import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='antialiased-cnns',  
     version='0.2',
     author="Richard Zhang",
     author_email="rizhang@adobe.com",
     description="Antialiased models and pooling layer from Zhang. Making Convnets Shift-Invariant Again. ICML 2019.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/adobe/antialiased-cnns",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: Other/Proprietary License",
         "Operating System :: OS Independent",
     ],
 )
