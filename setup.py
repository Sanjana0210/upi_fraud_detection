from setuptools import setup, find_packages

setup(
    name="upi_fraud_detection",
    version="1.0.0",
    description="UPI Fraud Detection System using Big Data Analytics",
    author="Hriday",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "imbalanced-learn>=0.12",
        "pyspark>=3.5",
        "pymongo>=4.7",
        "kafka-python>=2.0",
        "streamlit>=1.35",
        "plotly>=5.22",
        "python-dotenv>=1.0",
        "tqdm>=4.66",
    ],
    entry_points={
        "console_scripts": [
            "upi-insert-data=mongodb.insert_data:insert_data",
            "upi-train=ml.train_model:train",
            "upi-evaluate=ml.evaluate:evaluate",
        ]
    },
)
