import os
import math
import random
import glob
import numpy as np
from PIL import Image, ImageFont, ImageDraw


class Captcha:
    '''
    验证码。
    '''

    def __init__(self, font_size=28):
        '''
        初始化。
        '''

        font_path = os.path.dirname(__file__) + '/shs-cn-b.ttf'
        self.font_size = font_size
        self.font = ImageFont.truetype(
            font=font_path,
            size=font_size
        )
        self.noise_font = ImageFont.truetype(
            font=font_path,
            size=math.ceil(font_size * 0.3)
        )

    def save(self, text, path):
        '''
        生成并保存。
        '''

        with open(path, 'wb') as writer:
            image = self.draw(text)
            image.save(writer)

    def draw(self, text):
        '''
        绘制成图。
        '''

        length = len(text)
        height = math.ceil(self.font_size * 2)
        width = math.ceil(length * height * 0.6)
        size = (width, height)
        image = Image.new(mode="RGBA", size=size, color=(243, 251, 254, 255))
        image = self.draw_noise(image)
        return self.draw_text(image, text)

    def draw_text(self, image, text):
        '''
        绘制验证码
        '''

        color = self.random_rgba(255)
        for i in range(len(text)):
            left = int(self.font_size * self.vague_float(i, 0.2))
            code = text[i]
            item = self.draw_code(code, self.font, color)
            temp = Image.new(mode="RGBA", size=image.size)
            temp.paste(item, box=(left, 0))
            image = Image.alpha_composite(image, temp)
        return image

    def draw_noise(self, image):
        '''
        生成干扰背景字符。
        '''

        codeset = '2345678abcdefhijkmnpqrstuvwxyz'
        code_index_max = len(codeset) - 1
        w = image.size[0] - 1
        h = image.size[1] - 1
        for _ in range(100):
            code = codeset[random.randint(0, code_index_max)]
            ci = self.draw_code(code, self.noise_font, self.random_rgba())
            x = random.randint(0, w)
            y = random.randint(0, h)
            temp = Image.new(mode="RGBA", size=image.size)
            temp.paste(ci, box=(x, y))
            image = Image.alpha_composite(image, temp)
        return image

    @staticmethod
    def roll_text(length=6, codeset='2345678abcdefhijkmnpqrstuvwxyz'):
        '''
        随机生成。
        '''

        result = []
        code_index_max = len(codeset) - 1
        for _ in range(length):
            i = random.randint(0, code_index_max)
            result.append(codeset[i])
        return ''.join(result)

    @staticmethod
    def draw_code(code, font, color):
        '''
        绘制字符。
        '''

        w = int(font.size * 2)
        h = int(font.size * 2)
        size = (w, h)
        position = (
            math.ceil(w * 0.3),
            math.ceil(h * 0.05)
        )
        image = Image.new(mode="RGBA", size=size)
        draw = ImageDraw.Draw(image)
        draw.text(position, code, color, font=font)
        angle = random.randint(-50, 50)
        return image.rotate(angle, expand=1, resample=Image.BILINEAR)

    @staticmethod
    def vague_float(value, scope):
        '''
        让数在一定范围内模糊浮动。
        '''

        ps = random.random() * scope
        ns = random.random() * scope
        return value + ps - ns

    @staticmethod
    def random_rgba(alpth=None):
        '''
        随机的 RGBA 颜色。
        '''

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        a = random.randint(0, 255) if alpth == None else alpth
        return (r, g, b, a)


class CaptchaAssets:
    '''
    验证码数据集
    '''

    def __init__(self, folder, text_length=6, charset='2345678abcdefhijkmnpqrstuvwxyz'):
        '''
        初始化数据集。
        '''

        self.paths = glob.glob(f'{folder}/*.png')
        self.path_count = len(self.paths)
        self.path_max_index = self.path_count - 1
        self.text_length = text_length
        self.charset = charset
        self.charset_count = len(charset)
        self.charset_map = {}
        for i, c in enumerate(charset):
            self.charset_map[c] = i

    def get_one(self):
        '''
        随机加载一组数据。
        '''

        i = random.randint(0, self.path_max_index)
        p = self.paths[i]
        ir = np.array(Image.open(p))
        code = p.split('-')[1].split('.')[0]
        cr = np.zeros([self.text_length, self.charset_count])
        for i, c in enumerate(code):
            ci = self.charset_map[c]
            cr[i][ci] = 1
        return cr, ir

    def get_batch(self, batch_size=64):
        '''
        随机选取一组训练数据。
        '''

        code, image = self.get_one()
        x = np.zeros([batch_size, image.shape[0],
                      image.shape[1], image.shape[2]])
        y = np.zeros([batch_size, self.text_length, self.charset_count])

        for i in range(batch_size):
            code, image = self.get_one()
            x[i, :] = image
            y[i, :] = code
        return x, y
