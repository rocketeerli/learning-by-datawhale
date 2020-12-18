## 初识数据库

数据库（Database，DB）： 将大量数据保存起来，通过计算机加工而成的可以进行高效访问的数据集合。

数据库管理系统（Database Management System，DBMS）：用来管理数据库的计算机系统称为数据库管理系统。

### DBMS的种类

通过数据的保存格式（数据库的种类）来进行分类

- 层次数据库（Hierarchical Database，HDB）

- 关系数据库（Relational Database，RDB）

  - Oracle Database：甲骨文公司的RDBMS
  - SQL Server：微软公司的RDBMS
  - DB2：IBM公司的RDBMS
  - PostgreSQL：开源的RDBMS
  - MySQL：开源的RDBMS

  如上是5种具有代表性的RDBMS，其特点是由行和列组成的二维表来管理数据，这种类型的 DBMS 称为关系数据库管理系统（Relational Database Management System，RDBMS）。

- 面向对象数据库（Object Oriented Database，OODB）

- XML数据库（XML Database，XMLDB）

- 键值存储系统（Key-Value Store，KVS），举例：MongoDB

最常见的系统结构就是客户端 / 服务器类型（C/S类型）

## 初识 SQL

- 在数据库中，行称为**记录**，列称为**字段**。

- 完全基于标准 SQL 的 RDBMS 很少，通常需要根据不同的 RDBMS 来编写特定的 SQL 语句

### 三类语句

1. **DDL**（Data Definition Language，数据定义语言）：`CREATE`、 `DROP`、`ALTER`
2. **DML**（Data Manipulation Language，数据操纵语言）：`SELECT`、`INSERT`、`UPDATE`、`DELETE`
3. **DCL**（Data Control Language，数据控制语言）：`COMMIT`、`ROLLBACK`、`GRANT`、`REVOKE`

### 基本规则

* 以分号结尾

* 大小写

  - win 系统默认不区分表名及字段名的大小写

  - linux / mac 默认严格区分表名及字段名的大小写

- 创建数据库（ `CREATE DATABASE` 语句）

语法：

```sql
CREATE DATABASE < 数据库名称 > ;
```

创建本课程使用的数据库

```sql
CREATE DATABASE shop;
```

* 创建表（ `CREATE TABLE` 语句）

语法：

```sql
CREATE TABLE < 表名 >
( < 列名 1> < 数据类型 > < 该列所需约束 > ,
  < 列名 2> < 数据类型 > < 该列所需约束 > ,
  < 列名 3> < 数据类型 > < 该列所需约束 > ,
  < 列名 4> < 数据类型 > < 该列所需约束 > ,
  .
  .
  .
  < 该表的约束 1> , < 该表的约束 2> ,……);
```

创建本课程用到的商品表

```sql
CREATE TABLE product
(product_id CHAR(4) NOT NULL,
 product_name VARCHAR(100) NOT NULL,
 product_type VARCHAR(32) NOT NULL,
 sale_price INTEGER ,
 purchase_price INTEGER ,
 regist_date DATE ,
 PRIMARY KEY (product_id));
```

- 数据类型
  - `INTEGER` 
  - `CHAR` 存储定长字符串，达不到最大长度时，使用半角空格进行补足
  - `VARCHAR` 存储可变长度字符串（比较常用）
  - `DATE` 日期格式

* 约束
  - `NOT NULL`是非空约束，即该列必须输入数据。
  - `PRIMARY KEY`是主键约束，代表该列值都是唯一的。

* 删除和更新表

  - 删除表的语法： ```DROP TABLE < 表名 > ;```

  - 添加列： ```ALTER TABLE < 表名 > ADD COLUMN < 列的定义 >;```

  - 删除列：```ALTER TABLE < 表名 > DROP COLUMN < 列名 >;```

  - 清空表：```TRUNCATE TABLE TABLE_NAME;``` 比 `drop` 和 `delete` 更快。

  - 更新数据：

    基本语法：

    ```sql
    UPDATE <表名>
    SET <列名> = <表达式> [, <列名2>=<表达式2>...];  
    WHERE <条件>;  -- 可选，非常重要。
    ORDER BY 子句;  --可选
    LIMIT 子句; --可选
    ```

  - 插入数据

    基本语法：

    ```sql
    INSERT INTO <表名> (列1, 列2, 列3, ……) VALUES (值1, 值2, 值3, ……);  
    ```

#### 本课程用表插入数据sql如下：

```sql
START TRANSACTION;
INSERT INTO product VALUES('0001', 'T恤衫', '衣服', 1000, 500, '2009-09-20');
INSERT INTO product VALUES('0002', '打孔器', '办公用品', 500, 320, '2009-09-11');
INSERT INTO product VALUES('0003', '运动T恤', '衣服', 4000, 2800, NULL);
INSERT INTO product VALUES('0004', '菜刀', '厨房用具', 3000, 2800, '2009-09-20');
INSERT INTO product VALUES('0005', '高压锅', '厨房用具', 6800, 5000, '2009-01-15');
INSERT INTO product VALUES('0006', '叉子', '厨房用具', 500, NULL, '2009-09-20');
INSERT INTO product VALUES('0007', '擦菜板', '厨房用具', 880, 790, '2008-04-28');
INSERT INTO product VALUES('0008', '圆珠笔', '办公用品', 100, NULL, '2009-11-11');
COMMIT;
```

## 练习题

### 1.1 编写一条 CREATE TABLE 语句，用来创建一个包含表 1-A 中所列各项的表 Addressbook （地址簿），并为 regist_no （注册编号）列设置主键约束

```sql
CREATE TABLE Addressbook(
regist_no INTEGER NOT NULL COMMENT "注册编号",
`name` VARCHAR(128) NOT NULL COMMENT "姓名",
address VARCHAR(256) COMMENT "住址",
tel_no CHAR(10) COMMENT "电话号码",
mail_address CHAR(20) COMMENT "邮箱地址",
PRIMARY KEY(regist_no));
```

### 1.2 假设在创建练习1.1中的 Addressbook 表时忘记添加如下一列 postal_code （邮政编码）了，请把此列添加到 Addressbook 表中。

列名 ： postal_code

数据类型 ：定长字符串类型（长度为 8）

约束 ：不能为 NULL

```sql
ALTER TABLE addressbook ADD COLUMN postal_code CHAR(8) NOT NULL;
```

### 1.3 编写 SQL 语句来删除 Addressbook 表。

```SQL
DROP TABLE addressbook;
```

### 1.4 编写 SQL 语句来恢复删除掉的 Addressbook 表。

```SQL
CREATE TABLE Addressbook(
regist_no INTEGER NOT NULL COMMENT "注册编号",
`name` VARCHAR(128) NOT NULL COMMENT "姓名",
address VARCHAR(256) COMMENT "住址",
tel_no CHAR(10) COMMENT "电话号码",
mail_address CHAR(20) COMMENT "邮箱地址",
postal_code CHAR(8) NOT NULL,
PRIMARY KEY(regist_no));
```

