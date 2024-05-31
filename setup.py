from setuptools import setup, find_packages

setup(
    name='topicminer',
    version='0.1.0',
    description='A text analytics toolkit for processing and analyzing email data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aaron Noah Horvitz',
    author_email='AaronNHorvitz@gmail.com.com',
    url='https://github.com/AaronNHorvitz/topicminer',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'gensim',
        'pyLDAvis'
    ],
    python_requires='>=3.11',
)