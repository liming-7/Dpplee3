from setuptools import setup
from setuptools import find_packages

setup(name='dpplee3',
      version='0.1',
      description='Test dpplee3',
      url='http://github.com/uestcliming/Dpplee3',
      # download_url='https://github.com/maxpumperla/elephas/tarball/0.3',
      author='Lee ming',
      author_email='909640601@qq.com',
      install_requires=['pytorch', 'spark', 'flask'],
      license='MIT',
      packages=find_packages())
