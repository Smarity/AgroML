from setuptools import setup

setup(
   name='agroml',
   version='1.0',
   description='ML models to agroclimatological use',
   author='Juan Antonio Bellido',
   author_email='p22bejij@uco.es',
   packages=['agroml'],  #same as name
   install_requires=['icecream', 'hpelm', 'pandas', 
   'pytest', 'tensorflow', 'scikit_optimize', 'matplotlib', 
   'numpy', 'scikit_learn'], #external packages as dependencies
)