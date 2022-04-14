# Take Home Assignment

## Prerequisites

1. Apache Spark. Please see installation instructions [here](https://spark.apache.org/downloads.html).
2. Pyspark. To install just run `pip install pyspark`
3. FastAPI. To install just run `pip install fastapi uvicorn`

## Usage
1. Clone the repository: `git clone https://github.com/alexv71/spark_nlp_test`
2. `cd spark_nlp_test`
3. Execute `uvicorn main:app` and wait for full loading of the web service
4. Navigate your browser to http://localhost:8000/YOUR_SEARCH_PHRASE_HERE
5. If you need to get web service accessible from anywhere, execute `uvicorn main:app --host 0.0.0.0` and check the firewall settings

Examples:

http://localhost:8000/picnic%20bag

http://localhost:8000/babooshka

http://localhost:8000/retrospot%20umbrella

Note: Apache Spark should be installed and started on the local machine.

## Versions

Web service was tested on MS Windows 10 and Ubuntu Linux 18.04

Apache Spark: 3.1.3 for Hadoop 2.7

## Presentation

https://docs.google.com/presentation/d/1YysVm8qx869WbsyXq7dJdCPPggVbpMlbM6xuUXcu_As/edit?usp=sharing