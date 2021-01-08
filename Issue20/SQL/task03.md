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

- 例子

```sql
SELECT m,
ABS(m) AS abs_col ,
n, p,
MOD(n, p) AS mod_col,
ROUND(m,1)ASround_colS
FROM samplemath;
```

### 字符串函数

字符串函数也经常被使用，为了学习字符串函数，在此我们构造`samplestr`表。

```sql
-- DDL ：创建表
USE  shop;
DROP TABLE IF EXISTS samplestr;
CREATE TABLE samplestr
(str1 VARCHAR (40),
str2 VARCHAR (40),
str3 VARCHAR (40)
);
-- DML：插入数据
START TRANSACTION;
INSERT INTO samplestr (str1, str2, str3) VALUES ('opx',	'rt', NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES ('abc', 'def', NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES ('太阳',	'月亮', '火星');
INSERT INTO samplestr (str1, str2, str3) VALUES ('aaa',	NULL, NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES (NULL, 'xyz', NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES ('@!#$%', NULL, NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES ('ABC', NULL, NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES ('aBC', NULL, NULL);
INSERT INTO samplestr (str1, str2, str3) VALUES ('abc哈哈',  'abc', 'ABC');
INSERT INTO samplestr (str1, str2, str3) VALUES ('abcdefabc', 'abc', 'ABC');
INSERT INTO samplestr (str1, str2, str3) VALUES ('micmic', 'i', 'I');
COMMIT;
```

- CONCAT – 拼接

语法：`CONCAT(str1, str2, str3)`

- LENGTH – 字符串长度

语法：`LENGTH( 字符串 )`

- LOWER – 小写转换 （UPPER 函数用于大写转换。）

- REPLACE – 字符串的替换

语法：`REPLACE( 对象字符串，替换前的字符串，替换后的字符串 )`

- SUBSTRING – 字符串的截取

语法：`SUBSTRING （对象字符串 FROM 截取的起始位置 FOR 截取的字符数）`

- SUBSTRING_INDEX – 字符串按索引截取

语法：`SUBSTRING_INDEX (原始字符串， 分隔符，n)`

该函数用来获取原始字符串按照分隔符分割后，第 n 个分隔符之前（或之后）的子字符串，支持正向和反向索引，索引起始值分别为 1 和 -1。

- 例子

```sql
SELECT str1, str2, str3,
	CONCAT(str1, str2, str3) AS str_concat,
	LENGTH(str1) AS len_str1,
	LOWER(str1) AS low_str1,
	REPLACE(str1, str2, str3) AS rep_str,
	SUBSTRING(str1 FROM 3 FOR 2) AS sub_str,
	SUBSTRING_INDEX(str1, 'b', 1) AS sub_str_index
FROM samplestr;
```

### 日期函数

- CURRENT_DATE – 获取当前日期 

用法举例：`SELECT CURRENT_DATE;`，下同。

- CURRENT_TIME – 当前时间

- CURRENT_TIMESTAMP – 当前日期和时间

- EXTRACT – 截取日期元素

  语法：`EXTRACT(日期元素 FROM 日期)`

  使用 EXTRACT 函数可以截取出日期数据中的一部分，例如“年”、“月”或者“小时”“秒”等。该函数的返回值并不是日期类型而是数值类型

  - 例子：

  ```sql
  SELECT CURRENT_TIMESTAMP as now,
  EXTRACT(YEAR   FROM CURRENT_TIMESTAMP) AS year,
  EXTRACT(MONTH  FROM CURRENT_TIMESTAMP) AS month,
  EXTRACT(DAY    FROM CURRENT_TIMESTAMP) AS day,
  EXTRACT(HOUR   FROM CURRENT_TIMESTAMP) AS hour,
  EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) AS MINute,
  EXTRACT(SECOND FROM CURRENT_TIMESTAMP) AS second;
  ```

### 转换函数

在 SQL 中主要有两种转换：一是数据类型的转换，简称为类型转换，在英语中称为`cast`；另一个是值的转换。

- CAST – 类型转换

  - 语法：`CAST（转换前的值 AS 想要转换的数据类型）`

  - 例子：

  ```sql
  -- 将字符串类型转换为日期类型
  SELECT CAST('2021-01-6' AS DATE) AS date_col;
  ```

- COALESCE – 将NULL转换为其他值

  COALESCE 是 SQL 特有的函数。该函数会返回可变参数 A 中左侧开始第 1个不是NULL的值。

  - 语法：`COALESCE(数据1，数据2，数据3……)`
  - 例子：

  ```sql
  SELECT COALESCE(NULL, 11) AS col_1,
         COALESCE(NULL, 'hello world', NULL) AS col_2,
         COALESCE(NULL, NULL, '2020-11-01') AS col_3;
  ```

## 3.4 谓词

- 定义：谓词就是返回值为真值的函数。包括`TRUE / FALSE / UNKNOWN`。

- 分类：LIKE、BETWEEN、IS NULL、IS NOT NULL、IN、EXISTS

### LIKE 谓词 

用于字符串的部分一致查询，当需要进行字符串的部分一致查询时需要使用该谓词。

部分一致大体可以分为前方一致、中间一致和后方一致三种类型。

为了演示，首先需要创建一个表：

```sql
-- DDL ：创建表
CREATE TABLE samplelike
( strcol VARCHAR(6) NOT NULL,
PRIMARY KEY (strcol));
-- DML ：插入数据
START TRANSACTION; -- 开始事务
INSERT INTO samplelike (strcol) VALUES ('abcddd');
INSERT INTO samplelike (strcol) VALUES ('dddabc');
INSERT INTO samplelike (strcol) VALUES ('abdddc');
INSERT INTO samplelike (strcol) VALUES ('abcdd');
INSERT INTO samplelike (strcol) VALUES ('ddabc');
INSERT INTO samplelike (strcol) VALUES ('abddc');
COMMIT; -- 提交事务
SELECT * FROM samplelike;
```

- 前方一致：从字符串首字符开始匹配。（例如：选取出“dddabc”）

```sql
SELECT *
FROM samplelike
WHERE strcol LIKE 'ddd%';
```

其中的 `%` 是代表“零个或多个任意字符串”的特殊符号，本例中代表“以ddd开头的所有字符串”。

- 中间一致：字符串中间匹配。（例如：选取出“abcddd”, “dddabc”, “abdddc”）

```sql
SELECT *
FROM samplelike
WHERE strcol LIKE '%ddd%';
```

- 后方一致：从字符串最后开始匹配。（例如：选取出“abcddd“）

```sql
SELECT *
FROM samplelike
WHERE strcol LIKE '%ddd';
```

- `_`下划线匹配任意 1 个字符

使用 _（下划线）来代替 `%`，与 `%` 不同的是，它代表了“任意 1 个字符”。

```sql
SELECT *
FROM samplelike
WHERE strcol LIKE 'abc__';
```

### BETWEEN谓词

使用 BETWEEN 可以进行范围查询。该谓词与其他谓词或者函数的不同之处在于它使用了 3 个参数。

- BETWEEN 例子

```sql
-- 选取销售单价为100～ 1000元的商品
SELECT product_name, sale_price
FROM product
WHERE sale_price BETWEEN 100 AND 1000;
```

BETWEEN 的特点就是结果中会包含 100 和 1000 这两个临界值，也就是闭区间。

如果不想让结果中包含临界值，那就必须使用 < 和 >。

- 使用 < 和 > 例子

```sql
SELECT product_name, sale_price
FROM product
WHERE sale_price > 100
AND sale_price < 1000;
```

### IS NULL、 IS NOT NULL

- 为了选取出某些值为 NULL 的列的数据，不能使用 =，而只能使用特定的谓词IS NULL。

```sql
SELECT product_name, purchase_price
FROM product
WHERE purchase_price IS NULL;
```

- 想要选取 NULL 以外的数据时，需要使用IS NOT NULL。

```sql
SELECT product_name, purchase_price
FROM product
WHERE purchase_price IS NOT NULL;
```

### IN 谓词

`IN` 谓词是 `OR` 的简便用法。

- 多个查询条件取并集时可以选择使用`or`语句。

```sql
-- 通过OR指定多个进货单价进行查询
SELECT product_name, purchase_price
FROM product
WHERE purchase_price = 320
OR purchase_price = 500
OR purchase_price = 5000;
```

随着希望选取的对象越来越多， SQL 语句也会越来越长，阅读起来也会越来越困难。

- 使用 IN 谓词`IN(值1, 值2, 值3, …)` 来替换上述 SQL 语句。

```sql
SELECT product_name, purchase_price
FROM product
WHERE purchase_price IN (320, 500, 5000);
```

需要注意的是:

**在使用IN 和 NOT IN 时是无法选取出NULL数据的。NULL 只能使用 IS NULL 和 IS NOT NULL 来进行判断。**

- 使用子查询作为 IN 谓词的参数

首先创建表：

```sql
-- DDL ：创建表
DROP TABLE IF EXISTS shopproduct;
CREATE TABLE shopproduct
(  shop_id CHAR(4)     NOT NULL,
 shop_name VARCHAR(200) NOT NULL,
product_id CHAR(4)      NOT NULL,
  quantity INTEGER      NOT NULL,
PRIMARY KEY (shop_id, product_id) -- 指定主键
);
-- DML ：插入数据
START TRANSACTION; -- 开始事务
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000A', '东京', '0001', 30);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000A', '东京', '0002', 50);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000A', '东京', '0003', 15);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000B', '名古屋', '0002', 30);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000B', '名古屋', '0003', 120);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000B', '名古屋', '0004', 20);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000B', '名古屋', '0006', 10);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000B', '名古屋', '0007', 40);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000C', '大阪', '0003', 20);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000C', '大阪', '0004', 50);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000C', '大阪', '0006', 90);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000C', '大阪', '0007', 70);
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES ('000D', '福冈', '0001', 100);
COMMIT; -- 提交事务
SELECT * FROM shopproduct;
```

- 例子

取出大阪在售商品的销售单价。

```sql
-- step1：取出大阪门店的在售商品 `product_id`
SELECT product_id
FROM shopproduct
WHERE shop_id = '000C';
```

```sql
-- step2：取出大阪门店在售商品的销售单价 `sale_price`
SELECT product_name, sale_price
FROM product
WHERE product_id IN (SELECT product_id
  FROM shopproduct
                       WHERE shop_id = '000C');
```

子查询展开的结果：

```sql
-- 子查询展开后的结果
SELECT product_name, sale_price
FROM product
WHERE product_id IN ('0003', '0004', '0006', '0007');
```

注意：

1. **使用子查询即可保持 sql 语句不变**，极大提高了程序的可维护性，这是系统开发中需要重点考虑的内容。

2. NOT IN 同样支持子查询作为参数，用法和 in 完全一样。

### EXIST 谓词

EXIST（存在）谓词的主语是“记录”。EXIST 的左侧并没有任何参数。因为 EXIST 是只有 1 个参数的谓词。 所以，EXIST 只需要在右侧书写 1 个参数，该参数通常都会是一个子查询。

- 继续以 IN和子查询 中的示例，使用 EXIST 选取出大阪门店在售商品的销售单价。

```sql
SELECT product_name, sale_price
  FROM product AS p
 WHERE EXISTS (SELECT *
                 FROM shopproduct AS sp
                WHERE sp.shop_id = '000C'
                  AND sp.product_id = p.product_id);
```

由于通过条件“SP.product_id = P.product_id”将 product 表和 shopproduct表进行了联接，因此作为参数的是关联子查询。 EXIST 通常会使用关联子查询作为参数。

- 子查询中的 `SELECT *`

**由于 EXIST 只关心记录是否存在，因此返回哪些列都没有关系**。因此，使用下面的查询语句，查询结果也不会发生变化。

```sql
SELECT product_name, sale_price
  FROM product AS p
 WHERE EXISTS (SELECT 1 -- 这里可以书写适当的常数
                 FROM shopproduct AS sp
                WHERE sp.shop_id = '000C'
                  AND sp.product_id = p.product_id);
```

**可以把在 EXIST 的子查询中书写 `SELECT *` 当作 SQL 的一种习惯。**

- 使用NOT EXIST替换NOT IN

就像 EXIST 可以用来替换 IN 一样， NOT IN 也可以用NOT EXIST来替换。NOT EXIST 与 EXIST 相反，当“不存在”满足子查询中指定条件的记录时返回真（TRUE）。

## 3.5 CASE 表达式

CASE 表达式是在区分情况时使用的，这种情况的区分在编程中通常称为（条件）分支。

CASE表达式的语法分为**简单CASE表达式**和**搜索CASE表达式**两种。搜索CASE表达式包含简单CASE表达式的全部功能。

- 语法：

```nohighlight
CASE WHEN <求值表达式> THEN <表达式>
     WHEN <求值表达式> THEN <表达式>
     WHEN <求值表达式> THEN <表达式>
     .
     .
     .
ELSE <表达式>
END 
```

上述语句执行时，依次判断 when 表达式是否为真值，是则执行 THEN 后的语句，如果所有的 when 表达式均为假，则执行 ELSE 后的语句。
无论多么庞大的 CASE 表达式，最后也只会返回一个值。

- 例子

要实现如下结果：

```nohighlight
A ：衣服
B ：办公用品
C ：厨房用具  
```

1. 根据不同分支得到不同列值

```sql
SELECT  product_name,
        CASE WHEN product_type = '衣服' THEN CONCAT('A ： ',product_type)
             WHEN product_type = '办公用品'  THEN CONCAT('B ： ',product_type)
             WHEN product_type = '厨房用具'  THEN CONCAT('C ： ',product_type)
             ELSE NULL
        END AS abc_product_type
  FROM  product;
```

**ELSE 子句也可以省略不写，这时会被默认为 ELSE NULL。CASE 表达式最后的“END”是不能省略的。**

2. 实现列方向上的聚合

聚合函数 + CASE WHEN 表达式即可实现该效果

```sql
-- 对按照商品种类计算出的销售单价合计值进行行列转换
SELECT SUM(CASE WHEN product_type = '衣服' THEN sale_price ELSE 0 END) AS sum_price_clothes,
       SUM(CASE WHEN product_type = '厨房用具' THEN sale_price ELSE 0 END) AS sum_price_kitchen,
       SUM(CASE WHEN product_type = '办公用品' THEN sale_price ELSE 0 END) AS sum_price_office
  FROM product;
```

3. 实现行转列

```sql
-- CASE WHEN 实现数字列 score 行转列
SELECT name,
       SUM(CASE WHEN subject = '语文' THEN score ELSE null END) as chinese,
       SUM(CASE WHEN subject = '数学' THEN score ELSE null END) as math,
       SUM(CASE WHEN subject = '外语' THEN score ELSE null END) as english
  FROM score
 GROUP BY name;
```

- 当待转换列为数字时，可以使用`SUM AVG MAX MIN`等聚合函数；
- 当待转换列为文本时，可以使用`MAX MIN`等聚合函数

## 练习题

### 3.5 判断题

**Question:** 运算或者函数中含有 NULL 时，结果全都会变为NULL ？（判断题）

**Answer:** 正确。（？？？）

### 3.6 执行查询语句

**Question:** 对本章中使用的 product（商品）表执行如下 2 条 SELECT 语句，能够得到什么样的结果呢？

①

```sql
SELECT product_name, purchase_price
  FROM product
 WHERE purchase_price NOT IN (500, 2800, 5000);
```

②

```sql
SELECT product_name, purchase_price
  FROM product
 WHERE purchase_price NOT IN (500, 2800, 5000, NULL);
```

**Answer:** 

① 选出成本不是 500, 2800 和 5000 的商品名字和成本。

解析：该查询语句仅仅取出了 `purchase_price` 不是 500、2800、5000 的商品，而不包含 `purchase_price` 为 **`NULL`** 的商品，这是因为 **谓词无法与 `NULL` 进行比较**。

② 查询结果为空。（在使用IN 和 NOT IN 时是无法选取出NULL数据的。）

解析：代码执行之前，你可能会认为该语句会返回和查询 ① 同样的结果，实际上它却返回了零条记录，这是因为 **`NOT IN`** 的参数中不能包含 **`NULL`**，否则，查询结果通常为空。

### 3.7 编写 SQL 语句

**Question:**

按照销售单价（ sale_price）对练习 3.6 中的 product（商品）表中的商品进行如下分类。

- 低档商品：销售单价在1000日元以下（T恤衫、办公用品、叉子、擦菜板、 圆珠笔）
- 中档商品：销售单价在1001日元以上3000日元以下（菜刀）
- 高档商品：销售单价在3001日元以上（运动T恤、高压锅）

请编写出统计上述商品种类中所包含的商品数量的 SELECT 语句，结果如下所示。

执行结果

```sql
low_price | mid_price | high_price
----------+-----------+------------
        5 |         1 |         2
```

**Answer:**

```sql
SELECT 	
SUM(case when sale_price <= 1000 then 1 ELSE 0 END) AS low_price,
SUM(case when sale_price BETWEEN 1001 AND 3000 then 1 ELSE 0 END) AS mid_price,
SUM(case when sale_price > 3000 then 1 ELSE 0 END) AS high_price
FROM product;
```

