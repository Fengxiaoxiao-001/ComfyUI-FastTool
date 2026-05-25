import os
import math
import time
import numpy as np
import torch
from PIL import Image, PngImagePlugin
from concurrent.futures import ThreadPoolExecutor, as_completed


def to_int32(val):
    val = int(val) & 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val


def hash_key(s):
    if not s:
        return 0
    hash_val = 0
    for char in s:
        hash_val = to_int32(ord(char) + to_int32(to_int32(hash_val << 5) - hash_val))
    return abs(hash_val)


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def gilbert2d(width, height):
    coordinates = []
    if width >= height:
        generate2d(0, 0, width, 0, 0, height, coordinates)
    else:
        generate2d(0, 0, 0, height, width, 0, coordinates)
    return coordinates


def generate2d(x, y, ax, ay, bx, by, coordinates):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = sign(ax), sign(ay)
    dbx, dby = sign(bx), sign(by)

    if h == 1:
        for _ in range(w):
            coordinates.append((x, y))
            x += dax
            y += day
        return
    if w == 1:
        for _ in range(h):
            coordinates.append((x, y))
            x += dbx
            y += dby
        return

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2 += dax
            ay2 += day
        generate2d(x, y, ax2, ay2, bx, by, coordinates)
        generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coordinates)
    else:
        if (h2 % 2) and (h > 2):
            bx2 += dbx
            by2 += dby
        generate2d(x, y, bx2, by2, ax2, ay2, coordinates)
        generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coordinates)
        generate2d(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                   -bx2, -by2, -(ax - ax2), -(ay - ay2), coordinates)


def apply_obfuscation(img_tensor, password, is_decrypt):
    B, H, W, C = img_tensor.shape
    total_pixels = H * W

    curve = gilbert2d(W, H)
    flat_curve = np.array([y * W + x for x, y in curve], dtype=np.int32)

    user_seed = hash_key(password)
    base_offset = round(((math.sqrt(5) - 1) / 2) * total_pixels)
    offset = int((base_offset + user_seed) % total_pixels)
    if offset == 0:
        offset = 1

    shifted_curve = np.roll(flat_curve, -offset)

    out_tensor = torch.zeros_like(img_tensor)

    for b in range(B):
        img_np = img_tensor[b].cpu().numpy()
        img_flat = img_np.reshape(total_pixels, C)
        dst_flat = np.empty_like(img_flat)

        if not is_decrypt:
            dst_flat[shifted_curve] = img_flat[flat_curve]
        else:
            dst_flat[flat_curve] = img_flat[shifted_curve]

        out_tensor[b] = torch.from_numpy(dst_flat.reshape(H, W, C))

    return out_tensor


class XiaoxiaoBaseNode:
    IS_DECRYPT = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {"default": "output_xiaoxiao", "label": "输出路径"}),
                "password": ("STRING", {"default": "", "label": "密码"}),
                "strip_metadata": ("BOOLEAN", {"default": True, "label": "擦除元数据"}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 8, "label": "最大并发线程数"}),
                "max_preview": ("INT", {"default": 4, "min": 1, "max": 8, "label": "预览返回最大张数"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "input_folder": ("STRING", {"default": "", "label": "输入文件夹"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "xiaoxiao/image"

    def _clean_path(self, path):
        return path.strip().strip('"').strip("'") if path else ""

    def _save_image(self, tensor, folder, filename, keep_metadata=False, original_info=None):
        """增强版保存 - 真正控制是否保留元数据"""
        img_np = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        if keep_metadata and original_info:

            png_info = PngImagePlugin.PngInfo()
            for key, value in original_info.items():
                if isinstance(value, (str, bytes)):
                    png_info.add_text(key, value)
                elif isinstance(value, int):
                    png_info.add_text(key, str(value))

            pil_img.save(os.path.join(folder, filename), format="PNG",
                         compress_level=1, pnginfo=png_info)
        else:

            pil_img.save(os.path.join(folder, filename), format="PNG", compress_level=1)

    def process(self, output_folder, password, strip_metadata, max_workers=4, max_preview=4, image=None,
                input_folder=""):
        output_folder = self._clean_path(output_folder)
        input_folder = self._clean_path(input_folder)

        if not os.path.isabs(output_folder):
            output_folder = os.path.join(os.getcwd(), output_folder)
        os.makedirs(output_folder, exist_ok=True)

        action_prefix = "dec_" if self.IS_DECRYPT else "enc_"
        processed_list = []

        try:
            if image is not None:
                processed = apply_obfuscation(image, password, self.IS_DECRYPT)
                processed_list.append(processed)
                for i in range(processed.shape[0]):
                    filename = f"{action_prefix}node_{int(time.time() * 1000)}_{i}.png"
                    self._save_image(processed[i], output_folder, filename,
                                     keep_metadata=not strip_metadata)

            elif input_folder and os.path.isdir(input_folder):
                valid_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
                files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_ext]

                print(f"[Xiaoxiao] 发现 {len(files)} 张图片...")

                def process_single_file(fn):
                    try:
                        in_path = os.path.join(input_folder, fn)
                        with Image.open(in_path) as pil_img:
                            original_info = dict(pil_img.info) if not strip_metadata else None

                            img_tensor = torch.from_numpy(
                                np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
                            ).unsqueeze(0)

                            processed = apply_obfuscation(img_tensor, password, self.IS_DECRYPT)

                            save_name = f"{action_prefix}{os.path.splitext(fn)[0]}.png"
                            self._save_image(processed[0], output_folder, save_name,
                                             keep_metadata=not strip_metadata,
                                             original_info=original_info)

                            return processed, fn
                    except Exception as e:
                        print(f"[Xiaoxiao] 处理失败 {fn}：{e}")
                        return None, fn

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_single_file, fn): fn for fn in files}
                    for future in as_completed(futures):
                        result, fn = future.result()
                        if result is not None:
                            processed_list.append(result)


        except Exception as e:
            print(f"[Xiaoxiao] 严重错误: {e}")
        finally:
            if 'executor' in locals():
                executor.shutdown(wait=True, cancel_futures=True)

        if not processed_list:
            return (torch.zeros((1, 512, 512, 3)),)

        final_batch = torch.cat(processed_list, dim=0)
        if final_batch.shape[0] > max_preview:
            final_batch = final_batch[-max_preview:]

        return (final_batch,)


class XiaoxiaoEncrypt(XiaoxiaoBaseNode):
    IS_DECRYPT = False


class XiaoxiaoDecrypt(XiaoxiaoBaseNode):
    IS_DECRYPT = True


NODE_CLASS_MAPPINGS = {
    "XiaoxiaoEncrypt": XiaoxiaoEncrypt,
    "XiaoxiaoDecrypt": XiaoxiaoDecrypt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiaoxiaoEncrypt": "🔒 Xiaoxiao Encrypt",
    "XiaoxiaoDecrypt": "🔓 Xiaoxiao Decrypt"
}
