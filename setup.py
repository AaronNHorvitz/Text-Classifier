from setuptools import setup, find_packages

setup(
    name='topicminer',
    version='0.1.0',
    description='A text analytics toolkit for processing and analyzing email data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aaron Noah Horvitz',
    author_email='AaronNHorvitz@gmail.com',
    url='https://github.com/AaronNHorvitz/topicminer',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas>=1.2.0',
        'gensim>=4.0.0',
        'pyLDAvis>=3.0.0',
        'nltk>=3.5',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    keywords='text analytics, email processing, topic modeling, NLP',
)