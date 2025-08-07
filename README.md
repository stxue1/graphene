An implementation of Graphene: Packing and Dependency-aware Scheduling for Data-Parallel Clusters

#### Abstract

The modern landscape of cluster computing often deals with increasingly complex workloads characterized by DAG-structured jobs. One recent trace that highlights these complex characteristics is the 2018 Alibaba cluster trace. These modern workloads warrant a re-evaluation of existing algorithms to determine their efficacy in real-world scenarios. 
Graphene is a near-optimal algorithm for scheduling Directed Acyclical Graph (DAG) workflows. In its original evaluation with a synthetic workload, it achieved up to a 50\% reduction in job completion time compared to a critical path scheduler. However, Graphene bases its claims on synthetically generated trace data. Since real world trace data is now available, such as the Alibaba cluster traces, this allows researchers to answer more complicated questions in cluster scheduling. Therefore, we analyzed concerns like queueing delay in the Alibaba cluster, then re-evaluated Graphene on the Alibaba trace to demonstrate its efficacy on real-world trace data.
