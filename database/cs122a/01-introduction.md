# 01 Introduction

### 1. What is a Database System?

1. What is a database: A very large, integrated collection of data
2. Usually a model of a real-world enterprise
   1. Entities
   2. Relationships
3. What is a database management system \(DBMS\): A software system designed to store, manage, and provide access to one or more databases

### 2. DBMS over File Systems

1. Stage large datasets:
2. Special code needed
3. Protect data from inconsistency
4. Crash recovery
5. Security and access control

### 3. Evolution of DBMS:

1. Files: Manual Coding \(Byte streams\)
2. CODASYL/IMS: Early DBMS Technologies \(Records and pointers\)
3. Relational: Relational DB Systems \(Declarative approach\)

### 4. Why use a DBMS?

1. Data independence
2. Efficient data access
3. Reduced application development time
4. Data integrity and security
5. Uniform data administration
6. Concurrent access, recovery from crashes

### 5. Why Study Databases?

1. Shift from computation to information
2. Datasets increasing in diversity and volume
3. DBMS field encompass most of CS

### 6. Data Models

1. A data model is a collection of concepts for describing data
2. A schema is a description of a particular collection of data, using a given data model
3. The relation model is \(still\) the most widely used data model today
   1. Relation - basically a table with rows and \(named\) columns
   2. Schema - describes the tables and their columns

### 7. Levels of Abstraction

Many views of one conceptual \(logical\) schema and an underlying physical schema

1. Views describe how different users see the data.
2. Conceptual schema defines the logical structure of the database
3. Physical schema describes the files and indexes used under the covers

   ![](../../.gitbook/assets/image%20%28362%29.png)

### 8. Data Independence

1. Applications are insulated \(at multiple levels\) from how data is actually structured and stored, thanks to schema layering and high-level-queries
   1. Logical data independence: Protection from changes in the logical structure of data
   2. Physical data independence: Protection from changes in the physical structure of data
2. One of the most important benefits of DBMS use!

Allows changes to occur – w/o application rewrites!

### 9. Example: University DB

![](../../.gitbook/assets/image%20%28630%29.png)

![](../../.gitbook/assets/image%20%28625%29.png)

### 10. Concurrency and Recovery

1. Concurrent execution of user programs is essential to achieve good DBMS performance
2. Errors or crashes may occur during, or soon after, the execution of users' programs
3. DBMS answer: Users/programmers can pretend that they are using a reliable,….

### 11. Transaction: An execution of a DB Program

1. Key concept is transaction: An automatic sequence of database actions \(e.g., SQL operations on records\).
2. Each transaction, when run to completion, is expected leave the DB in a consistent state if the DB was consistent before it started to run.
   1. Users can specify simple integrity constraints on the data, and the DBMS will enforce these constraints.
   2. Beyond this, the DBMS is happily clueless about the data’s semantics \(e.g., how bank interest is computed\).
   3. Ensuring that any one transaction \(when run all by itself\) preserves consistency is the programmers job

### 12. Features of DBMS Transactions

1. Concurrency: The DBMS ensures that its execution of {T1, T2, …, Tn} is equivalent to some \(in fact, any\) serial execution.
   1. Before r/w a record, a transaction must request a lock on the record and wait until the DBMS grants it. \(All locks are released together, at the end of the transaction.\)
   2. Key Idea: If any action of a transaction Ti \(e.g., writing record X\) impacts Tj \(e.g., reading record X\), one of them will lock X first and the other will have to wait until the first one is done – which orders the transactions!
2. Atomicity: The DBMS ensures atomicity \(an all-or-nothing outcome\) even if the system crashes in the middle of a Xact.
   1. Idea: Keep a log \(history\) of all actions carried out by the DBMS while executing a set of Xacts:
      1. Before a change is made to the database, a log entry \(old value, new value\) is forced to a safe \(different\) location.
      2. In the event of a crash, the effects of partially executed transactions can first be undone using the log.
      3. In the event of a data loss following a successful finish, lost transaction effects can also be redone using the log.
      4. Note: The DBMS does all of this transparently!

### 13. DBMS Structure

![](../../.gitbook/assets/image%20%28345%29.png)

![](../../.gitbook/assets/image%20%28789%29.png)

### 14. Components' Roles

1. Query Parser
   1. Parse and analyze SQL query
   2. Make sure the query is valid and talking about tables, etc., that indeed exist
2. Query optimizer \(often with 2 steps\)
   1. Rewrite the query logically
   2. Perform cost-based optimization
   3. Goal is a "good" query plan considering
      1. Physical table structures
      2. Available access paths \(indexes\)
      3. Data statistics \(if known\)
      4. Cost model \(for relational operations\)
3. Plan Executor + Relational Operators
   1. Runtime side of query processing
   2. Query plan is a tree of relational operators \(drawn from the relational algebra\)

      ![](../../.gitbook/assets/image%20%28766%29.png)
4. Files of Records
   1. OSs usually have byte-stream based APIs
   2. DBMSs instead provide record based APIs
      1. Record = set of fields
      2. Fields are typed
      3. Records reside on pages of files
5. Access Methods
   1. Index structures for lookups based on field values
   2. We’ll look in more depth at B+ tree indexes in this class \(as they are the most commonly used indexes across all commercial and open source systems\)
6. Buffer Manager
   1. The DBMS answer to main memory management!
   2. All disk page accesses go through the buffer pool
   3. Buffer manager caches pages from files and indices
   4. “DB-oriented” page replacement scheme\(s\)
   5. Also interacts with logging \(so undo/redo possible\)
7. Disk Space and I/O Managers
   1. Manage space on disk \(pages\), including extents
   2. Also manage I/O \(sync, async, prefetch, …\)
   3. Remember: database data is persistent
8. System Catalog \(or “Metadata”\)
   1. Info about physical data \(file system stuff\)
   2. Info about tables \(name, columns, types, … \);
   3. also, info about any constraints, keys, etc.
   4. Data statistics \(e.g., value distributions, counts, …\)
   5. Info about indexes \(kinds, target tables, …\)
   6. And so on! \(Views, security, …\)
9. Transaction Management
   1. ACID \(Atomicity, Consistency, Isolation, Durability\)
   2. Lock Manager for Consistency + Isolation
   3. Log Manager for Atomicity + Durability

### 15. Miscellany: Some Terminology

1. Data Definition Language \(DDL\): Used to express views + logical schemas \(using a syntactic form of a a data model, e.g., relational\)
2. Data Manipulation Language \(DML\): Used to access and update the data in the database \(again in terms of a data model, e.g., relational\)
3. Query Language \(QL\): Synonym for DML or its retrieval \(i.e., data access or query\) sublanguage
4. Database Administrator \(DBA\): The “super user” for a database or a DBMS. Deals with things like physical DB design, tuning, performance monitoring, backup/restore, user and group authorization management.
5. Application Developer: Builds data-centric applications. Involved with logical DB design, queries, and DB application tools \(e.g., JDBC, ORM, …\)
6. Data Analyst or End User: Non-expert who uses tools to interact w/the data

### 16. A Brief History of DB

### 17. Summary

1. DBMS is used to maintain & query large datasets.
2. Benefits include recovery from system crashes, concurrent access, quick application development, data integrity and security.
3. Levels of abstraction give data independence.
4. A DBMS typically has a layered architecture.
5. DBAs \(and friends\) hold responsible jobs and they are also well-paid.
6. Data-related R&D is one of the broadest, most exciting areas in CS.

