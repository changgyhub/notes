# 03 Relational Database

### 1. Definitions

![](../../.gitbook/assets/image%20%2833%29.png)

### 2. Example Instance of Students Relation

![](../../.gitbook/assets/image%20%28158%29.png)

### 3. Relational Query Languages

1. A major strength of the relational model: supports simple, powerful querying of data.
2. Queries can be written intuitively, and the DBMS is responsible for efficient evaluation.
   * The key: precise \(and set-based\) semantics for relational queries.
   * Allows the optimizer to extensively re-order operations, and still ensure that the answer does not change.

### 4. Preview of The SQL Query Language

![](../../.gitbook/assets/image%20%28768%29.png)

### 5. Querying Multiple Relations

![](../../.gitbook/assets/image%20%28390%29.png)

### 6. Creating Relations in SQL

![](../../.gitbook/assets/image%20%28675%29.png)

### 7. Destroying and Altering Relations

![](../../.gitbook/assets/image%20%28352%29.png)

### 8. Adding and Deleting Tuples

![](../../.gitbook/assets/image%20%28449%29.png)

### 9. Integrity Constraints \(ICs\)

![](../../.gitbook/assets/image%20%28801%29.png)

### 10. Primary Key Constraints

![](../../.gitbook/assets/image%20%28639%29.png)

### 11. Primary and Candidate Keys in SQL

![](../../.gitbook/assets/image%20%28466%29.png)

### 12. Foreign Keys, Referential Integrity

![](../../.gitbook/assets/image%20%28497%29.png)

### 13. Foreign Keys in SQL

![](../../.gitbook/assets/image%20%28697%29.png)

### 14. Enforcing Referential Integrity

![](../../.gitbook/assets/image%20%28291%29.png)

### 15. Referential Integrity in SQL

![](../../.gitbook/assets/image%20%2822%29.png)

### 16. Where do ICs Come From?

![](../../.gitbook/assets/image%20%28823%29.png)

### 17. Logical DB Design: ER to Relational

![](../../.gitbook/assets/image%20%28476%29.png)

### 18. Relationship Sets to Tables

![](../../.gitbook/assets/image%20%28405%29.png)

### 19. Translating ER Diagrams with Key Constraints

![](../../.gitbook/assets/image%20%28367%29.png)

![](../../.gitbook/assets/image%20%28810%29.png)

### 20. Translating ER Diagrams with Participation Constraints

![](../../.gitbook/assets/image%20%28139%29.png)

![](../../.gitbook/assets/image%20%28446%29.png)

Note: we **cannot** enforce a many-to-many with total participation constraint in SQL, however, if there is a total participation, the other side should be **ON DELETE CASCADE**

### **21.** Translating ER Diagrams with Weak Entities

![](../../.gitbook/assets/image%20%28307%29.png)

![](../../.gitbook/assets/image%20%28746%29.png)

### 22. Translating ER Diagrams with ISA Hierarchies

![](../../.gitbook/assets/image%20%28462%29.png)

Approaches

![](../../.gitbook/assets/image%20%28691%29.png)

Options

![](../../.gitbook/assets/image%20%28592%29.png)

Considerations

1. Query convenience \(e.g. List the names of all Emps in lot 12A\)
2. PK enforcement \(e.g. Make sure that ssn is unique for all Emps\)
3. Relationship targets \(e.g. Lawyers table REFERENCES Contract\_Emps\)
4. Handling of overlap constraints \(e.g. Sally is under a contract for her hourly work\)
5. Space and query performance tradeoffs \(e.g. List all the info about hourly employee 123; What if most employees are “just plain employees”?\)

### 23. Mapping Advanced ER Features

![](../../.gitbook/assets/image%20%28686%29.png)

### 24. SQL Views \(and Security\)

![](../../.gitbook/assets/image%20%28701%29.png)

![](../../.gitbook/assets/image%20%28610%29.png)

### 25. Binary vs. Ternary Relationships

![](../../.gitbook/assets/image%20%28533%29.png)

* The key constraints let us combine Purchaser with Policies and Beneficiary with Dependents.
* Participation constraints lead to NOT NULL constraints.
  * Note: Primary key attributes are all NOT NULL as well – check documentation to see if that’s implicit or explicit!

### 26. Example

![](../../.gitbook/assets/image%20%28234%29.png)

![](../../.gitbook/assets/image%20%28683%29.png)

### 27. Summary

![](../../.gitbook/assets/image%20%28708%29.png)

