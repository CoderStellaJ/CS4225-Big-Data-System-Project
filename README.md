# MapReduce-based Federated Learning for Landmark Recognition

## Parts of Spark
![alt text](./imgs/spark_structure.png)

* Partitions 
<br/> A partition is a small chunk of a large distributed data set. 
Spark manages data using partitions that helps parallelize data processing with minimal data shuffle across the executors.

* Task 
<br/> A task is a unit of work that can be run on a partition of a distributed dataset and gets executed on a single executor. 
The unit of parallel execution is at the task level.
All the tasks with-in a single stage can be executed in parallel

* Executor 
<br/> An executor is a single JVM process which is launched for an application on a worker node. 
Executor runs tasks and keeps data in memory or disk storage across them. 
Each application has its own executors. A single node can run multiple executors 
and executors for an application can span multiple worker nodes. 
An executor stays up for the duration of the Spark Application and runs the tasks in multiple threads. 
The number of executors for a spark application can be specified inside the SparkConf 
or via the flag –num-executors from command-line.

## References
[Spark structure post](http://site.clairvoyantsoft.com/understanding-resource-allocation-configurations-spark-application/)