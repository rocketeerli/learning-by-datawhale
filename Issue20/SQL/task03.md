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