from setuptools import setup

setup(name='multibubs',
      version='0.1',
      description='Generalized Sup ADF test for Multiple Bubbles in a Time Series',
      url='https://github.com/mfjackson/multibubs',
      author='Marty Jackson',
      author_email='martin.jackson.ds@gmail.com',
      license='MIT',
      packages=['multibubs'],
      include_package_data='True',
      install_requires=['numpy', 'statsmodels'])
