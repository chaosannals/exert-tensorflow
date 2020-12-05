import os
import time
import math
from importlib import import_module
from loguru import logger
Captcha = import_module('data').Captcha

def random_text():
    '''
    '''


if __name__ == '__main__':
    captcha = Captcha()
    d = 'captcha'
    if not os.path.isdir(d):
        os.makedirs(d, 0o777, True)
    for i in range(100):
        t = math.floor(time.time())
        c = captcha.roll_text()
        p = f'{d}/{t}-{c}.png'
        captcha.save(c, p)
        logger.info(f'{c} => {p}')

