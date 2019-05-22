# 02 Conceptual DB Design

### 1. ER Model Basics

![](../../.gitbook/assets/image%20%28627%29.png)

![](../../.gitbook/assets/image%20%28366%29.png)

### 2. Cardinality Constraints

![](../../.gitbook/assets/image%20%28767%29.png)

### 3. Participation Constraints

![](../../.gitbook/assets/image%20%28536%29.png)

### 4. ER Model Example

![](../../.gitbook/assets/image%20%28650%29.png)

![](../../.gitbook/assets/image%20%28749%29.png)

### 5. Weak Entities

![](../../.gitbook/assets/image%20%28734%29.png)

### 6. Ternary Relationships \(and beyond\)

![](../../.gitbook/assets/image%20%28546%29.png)

### 7. ISA \(“is a”\) Hierarchies

![](../../.gitbook/assets/image%20%28795%29.png)

### 8. Aggregation

![](../../.gitbook/assets/image%20%28204%29.png)

### 9. Additional Advanced ER Features

![](../../.gitbook/assets/image%20%28418%29.png)

### 10. Conceptual Design Using the ER Model

1. Design Choices
   * Should a given concept be modeled as an entity or an attribute
   * Should a given concept be modeled as an entity or a relationship?
   * Characterizing relationships: Binary or ternary? Aggregation?
2. Constraints in the ER Model
   * A lot of data semantics can \(and should\) be captured.
   * But, not all constraints cannot be captured by ER diagrams. \(e.g. Department heads from earlier\)

### 11. Entity vs. Attribute

Question: Should address be an attribute of Employees or an entity \(connected to Employees by a relationship\)? 

Answer: Depends how we want to use address information, the data semantics, and also the model features:

1. If we have several addresses per employee, address must be an entity if we stick only to basic E-R concepts \(as attributes cannot be set-valued w/o advanced modeling goodies\).
2. If the structure \(city, street, etc.\) is important, e.g., we want to retrieve employees in a given city, address must be modeled as an entity \(since attribute values are atomic\) w/o advanced modeling goodies.
3. If the address itself is logically separate \(e.g., the property that’s located there\) and refer-able, it’s rightly an entity in any case.

Example:

![](../../.gitbook/assets/image%20%28758%29.png)

### 12. Entity vs. Relationship

![](../../.gitbook/assets/image%20%28248%29.png)

### 13. Binary vs. Ternary Relationships

![](../../.gitbook/assets/image%20%28845%29.png)

1. Example A: This example illustrated a case when two binary relationships were “better” than one ternary relationship.
2. Example B: An example in the other direction: a ternary relation Contracts relates entity sets Parts, Departments and Suppliers, and has descriptive attribute qty. No combination of binary relationships is an adequate substitute:
   * S “can-supply” P, D “needs” P, and D “deals-with” S does not imply that D has agreed to buy P from S.
   * And also, how/where else would we record qty?

![](../../.gitbook/assets/image%20%2829%29.png)

### 14. Database Design Process \(Flow\)

1. Requirements Gathering \(interviews\)
2. Conceptual Design \(using ER\)
3. Platform Choice \(which DBMS?\)
4. Logical Design \(for target data model\)
5. Physical Design \(for target DBMS, workload\)
6. Implement \(and test, of course J\)

### 15. Summary of Conceptual Design

1. Conceptual design follows requirements analysis
   * Yields a high-level description of data to be stored
2. ER model popular for conceptual design
   * Constructs are expressive, close to the way people think about their applications.
3. Basic constructs: entities, relationships, and attributes \(of entities and relationships\).
4. Additionally: weak entities, ISA hierarchies, aggregation, and multi-valued, composite and/or derived attributes.
5. Note: Many variations on the ER model \(and many notations in use as well\) – and also, UML
6. Several kinds of integrity constraints can be expressed in the ER model: cardinality constraints, participation constraints, also overlap/covering constraints for ISA hierarchies. Some foreign key constraints are also implicit in the definition of a relationship set \(more about key constraints will be coming soon\).
   * Some constraints \(notably, functional dependencies\) cannot be expressed in the ER model.
   * Constraints play an important role in determining the best database design for an enterprise.
7. ER design is subjective. There are often many ways to model a given scenario. Analyzing alternatives can be tricky, especially for a large enterprise. Common choices include:
   * Entity vs. attribute, entity vs. relationship, binary or n-ary relationship, whether or not to use an ISA hierarchy, and whether or not to use aggregation.
8. Ensuring good database design: The resulting relational schema should be analyzed and refined further. For this, FD information and normalization techniques are especially useful \(coming soon\).

