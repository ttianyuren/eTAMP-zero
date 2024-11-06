# Get Started with eTAMP and Symbolic Planning

This guide will help you get hands-on experience with the **eTAMP** (Extended Tree Search for Task and Motion Planning) algorithm and its symbolic planning backbone. eTAMP relies on the [**symk** planning algorithm](https://github.com/speckdavid/symk) to generate candidate high-level symbolic plans, which serve as a foundation for later detailed planning. In this guide, you'll explore symbolic planning using **PDDL** and see how these components interact within eTAMP.

Follow the steps below to explore symbolic planning concepts and experiment with the eTAMP environment.

---

## Step 1: Explore the Basics of Symbolic Planning

1. **Learn What Symbolic Planning Is**:
   - Symbolic planning is a process where high-level actions are sequenced to achieve a goal based on logical rules and relationships.
   - The **Planning Domain Definition Language (PDDL)** is commonly used to define the actions, objects, and rules in symbolic planning problems. eTAMP relies on **PDDL 2.2 or later**, which includes support for derived predicates, action costs, and equality.

2. **Watch an Introductory Tutorial**:
   - To start, watch this tutorial series on symbolic planning fundamentals: [AI Planning Video Tutorial](https://www.youtube.com/watch?v=7Vy8970q0Xc&list=PLwJ2VKmefmxpUJEGB1ff6yUZ5Zd7Gegn2).
   - This provides a gentle introduction to planning concepts and will help you understand the structure and logic of PDDL files used in eTAMP.

---

## Step 2: Set Up Your eTAMP Environment

1. **Clone the eTAMP Repository**:
   - Use this repository to start experimenting with eTAMP: [eTAMP-zero GitHub Repo](https://github.com/ttianyuren/eTAMP-zero).
   - After cloning, ensure all dependencies are installed by following any setup instructions in the repository.

2. **Locate the `symk` Script**:
   - In eTAMP, the **symk** algorithm is run through a script at [`connect_topk.py`](https://github.com/ttianyuren/eTAMP-zero/blob/main/connect_topk.py), which internally leverages **Fast Downward** to perform symbolic planning.
   - We’ll experiment with symk’s inputs (two PDDL files) to observe how changes affect the resulting symbolic plans.

---

## Step 3: Experiment with `symk` and PDDL Inputs

1. **Run the `symk` Algorithm**:
   - Open the script [connect_topk.py](https://github.com/ttianyuren/eTAMP-zero/blob/5b3cf89801ef435fc4545a823dabf872dfa06747/connect_topk.py#L92).
   - Notice the PDDL file paths: `"/pddl/domain.pddl"` and `"/pddl/problem.pddl"`. These are the inputs to `symk` for generating symbolic plans.

2. **Test Different PDDL Files**:
   - Swap in different PDDL files to see how they impact the output plans. You can find example PDDL files in [this Fast Downward benchmarks folder](https://github.com/aibasel/downward/tree/main/misc/tests/benchmarks).
   - Try modifying small parts of the PDDL files, such as adding or removing actions or constraints, to see how they impact the plan structure.
   - Run the script and observe how the output plan changes with different domains and problems. This will give you a practical sense of how symbolic planners interpret and generate plans.

---

## Step 4: Understand PDDL Requirements for eTAMP

1. **PDDL Version**:
   - **symk** and eTAMP rely on **PDDL 2.2**, which introduced important features like derived predicates, action costs, and equality. These additions allow for greater expressiveness in defining complex planning problems.
   
2. **Compatibility Note**:
   - symk only supports up to PDDL 2.2 because of its dependency on **Fast Downward** (which also supports PDDL 2.2). This means you don’t need to worry about later versions, which may have overly complex syntax and features that are not applicable in eTAMP.

---

## Step 5: Know what Fast Downward is

1. **Understand Fast Downward’s Role in eTAMP**:
   - **Fast Downward** is a powerful planning engine used by `symk` to explore possible symbolic plans. It’s flexible and supports a range of heuristics, though you don’t need to dive into all its details for task and motion planning (TAMP).
   - Fast Downward helps generate symbolic plan "skeletons" based on input PDDL files, which eTAMP then refines during its motion planning phase.

---

By following these steps, you'll gain practical knowledge of symbolic planning, become familiar with PDDL, and understand how eTAMP leverages symbolic plans for task and motion planning. Happy experimenting!