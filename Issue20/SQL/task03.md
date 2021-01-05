# 复杂一点的查询

## 3.1 视图

### 定义

**视图是一个虚拟的表，不同于直接操作数据表，视图是依据SELECT语句来创建的。**所以操作视图时会根据创建视图的SELECT语句生成一张虚拟表，然后在这张虚拟表上做SQL操作。

### 语法

```sql
SELECT stu_name FROM view_students_info;
```

单从表面上看起来这个语句是和正常的从数据表中查询数据是完全相同的，但其实我们操作的是一个视图。所以从SQL的角度来说操作视图与操作表看起来是完全相同的。

### 视图与表有什么区别

《sql基础教程第2版》用一句话非常凝练的概括了视图与表的区别—“是否保存了实际的数据”。即视图是基于真实表的一张虚拟的表，其数据来源均建立在真实表的基础上。

视图与表的关系：**“视图不是表，视图是虚表，视图依赖于表”**。

- 视图存在的必要性

1. 可以将频繁使用的SELECT语句保存以提高效率。
2. 可以使用户看到的数据更加清晰。
3. 可以不对外公开数据表全部字段，增强数据的保密性。
4. 可以降低数据的冗余。

### 创建视图

创建视图的基本语法如下：

```sql
CREATE VIEW <视图名称>(<列名1>,<列名2>,...) AS <SELECT语句>
```

其中SELECT 语句需要书写在 AS 关键字之后。 SELECT 语句中列的排列顺序和视图中列的排列顺序相同。

需要注意的是：

1. **视图名在数据库中需要是唯一的，不能与其他视图和表重名**。
2. **我们也可以在视图的基础上继续创建视图**。（尽量避免，因为对多数 DBMS 来说， 多重视图会降低 SQL 的性能。）
3. **在一般的DBMS中定义视图时不能使用ORDER BY语句**。（ MySQL中视图的定义允许使用ORDER BY语句）

#### 创建单表视图

我们在product表的基础上创建一个视图，如下：

```sql
CREATE VIEW productsum (product_type, cnt_product)
AS
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type ;
```

#### 创建多表视图

为了学习多表视图，我们再创建一张表，相关代码如下：

```sql
CREATE TABLE shop_product
(shop_id    CHAR(4)       NOT NULL,
 shop_name  VARCHAR(200)  NOT NULL,
 product_id CHAR(4)       NOT NULL,
 quantity   INTEGER       NOT NULL,
 PRIMARY KEY (shop_id, product_id));

INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000A',	'东京',		'0001',	30);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000A',	'东京',		'0002',	50);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000A',	'东京',		'0003',	15);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000B',	'名古屋',	'0002',	30);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000B',	'名古屋',	'0003',	120);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000B',	'名古屋',	'0004',	20);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000B',	'名古屋',	'0006',	10);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000B',	'名古屋',	'0007',	40);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000C',	'大阪',		'0003',	20);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000C',	'大阪',		'0004',	50);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000C',	'大阪',		'0006',	90);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000C',	'大阪',		'0007',	70);
INSERT INTO shop_product (shop_id, shop_name, product_id, quantity) VALUES ('000D',	'福冈',		'0001',	100);
```

我们在product表和shop_product表的基础上创建视图。

```sql
CREATE VIEW view_shop_product(product_type, sale_price, shop_name)
AS
SELECT product_type, sale_price, shop_name
  FROM product,
       shop_product
 WHERE product.product_id = shop_product.product_id;
```

我们可以在这个视图的基础上进行查询

```sql
SELECT sale_price, shop_name
  FROM view_shop_product
 WHERE product_type = '衣服';
```

### 修改视图结构

修改视图结构的基本语法如下：

```sql
ALTER VIEW <视图名> AS <SELECT语句>
```

我们修改上方的productSum视图为

```sql
ALTER VIEW productSum
    AS
        SELECT product_type, sale_price
          FROM Product
         WHERE regist_date > '2009-09-11';
```

### 更新视图内容

对于一个视图来说，如果包含以下结构的任意一种都是不可以被更新的：（？？？不是很明白）

- 聚合函数 SUM()、MIN()、MAX()、COUNT() 等。
- DISTINCT 关键字。
- GROUP BY 子句。
- HAVING 子句。
- UNION 或 UNION ALL 运算符。
- FROM 子句中包含多个表。

如果原表可以更新，那么视图中的数据也可以更新。

反之亦然，如果视图发生了改变，而原表没有进行相应更新的话，就无法保证数据的一致性了。

注意：**因为视图的定义，视图只是原表的一个窗口，所以它修改也只能修改透过窗口能看到的内容。在创建视图时尽量使用限制不允许通过视图来修改表**

### 删除视图

删除视图的基本语法如下：

```sql
DROP VIEW <视图名1> [ , <视图名2> …]
```

注意：**需要有相应的权限才能成功删除**。

删除刚才创建的 `productSum` 视图

```sql
DROP VIEW productSum;
```

## 3.2 子查询

### 定义

子查询指一个查询语句嵌套在另一个查询语句内部的查询，这个特性从 MySQL 4.1 开始引入，在 SELECT 子句中先计算子查询。

- 和视图的关系

子查询就是将用来定义视图的 SELECT 语句直接用于 FROM 子句当中。

### 子查询嵌套

虽然嵌套子查询可以查询出结果，但是随着子查询嵌套的层数的叠加，SQL语句不仅会难以理解而且执行效率也会很差，所以要尽量避免使用。

### 标量子查询

- 定义

所谓标量就是要求我们执行的SQL语句只能返回一个值，也就是要返回表中具体的**某一行的某一列**。

**标量子查询可以返回一个值**。

- 举例

如何通过标量子查询语句查询出销售单价高于平均销售单价的商品。

```sql
SELECT product_id, product_name, sale_price
  FROM product
 WHERE sale_price > (SELECT AVG(sale_price) FROM product);
```

由于标量子查询的特性，导致标量子查询不仅仅局限于 WHERE 子句中，通常任何可以使用单一值的位置都可以使用。也就是说， 能够使用常数或者列名的地方，无论是 SELECT 子句、GROUP BY 子句、HAVING 子句，还是 ORDER BY 子句，几乎所有的地方都可以使用。

- 重复列的行

还可以这样使用标量子查询：

```sql
SELECT product_id,
       product_name,
       sale_price,
       (SELECT AVG(sale_price)
          FROM product) AS avg_price
  FROM product;
```

如果直接使用 `AVG(sale_price)`， 仅输出一行数据；如果使用子查询，子查询出来的单个数据自动与每一个父查询的结果结合起来。

### 关联子查询

通过一系列操作，将子查询与主查询连接起来。例如，为内外子查询的表分别进行命名。

## 练习题

### 3.1 创建视图

创建出满足下述三个条件的视图（视图名称为 ViewPractice5_1）。使用 product（商品）表作为参照表，假设表中包含初始状态的 8 行数据。

- 条件 1：销售单价大于等于 1000 日元。
- 条件 2：登记日期是 2009 年 9 月 20 日。
- 条件 3：包含商品名称、销售单价和登记日期三列。

**Solution：**

```sql
CREATE VIEW ViewPractice5_1(product_name, sale_price, regist_date) AS
SELECT product_name, sale_price, regist_date FROM product WHERE (
	sale_price >= 1000 AND regist_date = '2009-09-20'
);
```

### 3.2 向视图中插入数据

向习题一中创建的视图  `ViewPractice5_1` 中插入如下数据，会得到什么样的结果呢？

**Solution:**

报错：

```
Field of view 'shop.viewpractice5_1' underlying table doesn't have a default value
```

### 3.3 编写查询语句

请根据如下结果编写 SELECT 语句，其中 `sale_price_all` 列为全部商品的平均销售单价。

```sql
product_id | product_name | product_type | sale_price | sale_price_all
------------+-------------+--------------+------------+---------------------
0001       | T恤衫         | 衣服         | 1000       | 2097.5000000000000000
0002       | 打孔器        | 办公用品      | 500        | 2097.5000000000000000
0003       | 运动T恤       | 衣服          | 4000      | 2097.5000000000000000
0004       | 菜刀          | 厨房用具      | 3000       | 2097.5000000000000000
0005       | 高压锅        | 厨房用具      | 6800       | 2097.5000000000000000
0006       | 叉子          | 厨房用具      | 500        | 2097.5000000000000000
0007       | 擦菜板        | 厨房用具       | 880       | 2097.5000000000000000
0008       | 圆珠笔        | 办公用品       | 100       | 2097.5000000000000000
```

**Solution:**

```sql
SELECT product_id, product_name, product_type, sale_price, 
(SELECT AVG(sale_price) FROM product) as sale_price_all
FROM product;
```

### 3.4 创建视图

请根据习题一中的条件编写一条 SQL 语句，创建一幅包含如下数据的视图（名称为 `AvgPriceByType`）。

```sql
product_id | product_name | product_type | sale_price | avg_sale_price
------------+-------------+--------------+------------+---------------------
0001       | T恤衫         | 衣服         | 1000       |2500.0000000000000000
0002       | 打孔器         | 办公用品     | 500        | 300.0000000000000000
0003       | 运动T恤        | 衣服        | 4000        |2500.0000000000000000
0004       | 菜刀          | 厨房用具      | 3000        |2795.0000000000000000
0005       | 高压锅         | 厨房用具     | 6800        |2795.0000000000000000
0006       | 叉子          | 厨房用具      | 500         |2795.0000000000000000
0007       | 擦菜板         | 厨房用具     | 880         |2795.0000000000000000
0008       | 圆珠笔         | 办公用品     | 100         | 300.0000000000000000
```

提示：其中的关键是 avg_sale_price 列。与习题三不同，这里需要计算出的 是各商品种类的平均销售单价。这与使用关联子查询所得到的结果相同。 也就是说，该列可以使用关联子查询进行创建。问题就是应该在什么地方使用这个关联子查询。

**Solution:**

```sql
CREATE VIEW AvgPriceByType AS
SELECT product_id, product_name, product_type, sale_price, 
(SELECT AVG(sale_price) FROM product AS p2 
WHERE p1.product_type = p2.product_type) as avg_sale_price
FROM product AS p1;
```

## 3.3 各种各样的函数

### 算数函数

- `+ - * /`四则运算

为了演示其他的几个算数函数，在此构造`samplemath`表

```sql
-- DDL ：创建表
USE shop;
DROP TABLE IF EXISTS samplemath;
CREATE TABLE samplemath
(m float(10,3),
n INT,
p INT);

-- DML ：插入数据
START TRANSACTION; -- 开始事务
INSERT INTO samplemath(m, n, p) VALUES (500, 0, NULL);
INSERT INTO samplemath(m, n, p) VALUES (-180, 0, NULL);
INSERT INTO samplemath(m, n, p) VALUES (NULL, NULL, NULL);
INSERT INTO samplemath(m, n, p) VALUES (NULL, 7, 3);
INSERT INTO samplemath(m, n, p) VALUES (NULL, 5, 2);
INSERT INTO samplemath(m, n, p) VALUES (NULL, 4, NULL);
INSERT INTO samplemath(m, n, p) VALUES (8, NULL, 3);
INSERT INTO samplemath(m, n, p) VALUES (2.27, 1, NULL);
INSERT INTO samplemath(m, n, p) VALUES (5.555,2, NULL);
INSERT INTO samplemath(m, n, p) VALUES (NULL, 1, NULL);
INSERT INTO samplemath(m, n, p) VALUES (8.76, NULL, NULL);
COMMIT; -- 提交事务
-- 查询表内容
SELECT * FROM samplemath;
```

- ABS – 绝对值

语法：`ABS( 数值 )`

- MOD – 求余数

语法：`MOD( 被除数，除数 )`、

注意：主流的 DBMS 都支持 MOD 函数，只有SQL Server 不支持该函数，其使用`%`符号来计算余数。

- ROUND – 四舍五入

语法：`ROUND( 对象数值，保留小数的位数 )`

- 练习

```sql
SELECT m,
ABS(m) AS abs_col ,
n, p,
MOD(n, p) AS mod_col,
ROUND(m,1)ASround_colS
FROM samplemath;
```

### 字符串函数