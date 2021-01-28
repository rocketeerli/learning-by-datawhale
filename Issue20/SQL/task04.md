# 集合运算

## 4.1 表的加减法

### 4.1.1 集合运算

在标准 SQL 中，分别对检索结果使用 `UNION`，`INTERSECT`， `EXCEPT` 来将检索结果进行并，交和差运算，像这种用来进行集合运算的运算符称为集合运算符。

### 4.1.2 表的加法——UNION

- UNION

  - 用法：

  ```sql
  SELECT product_id, product_name
    FROM product
   UNION
  SELECT product_id, product_name
    FROM product2;
  ```

  - 练习：

  假设连锁店想要增加毛利率超过 50%或者售价低于 800 的货物的存货量, 请使用 UNION 对分别满足上述两个条件的商品的查询结果求并集.

  ```
  -- 参考答案:
  SELECT  product_id,product_name,product_type
         ,sale_price,purchase_price
    FROM product 
   WHERE sale_price<800
    
   UNION
   
  SELECT  product_id,product_name,product_type
         ,sale_price,purchase_price
    FROM product 
   WHERE sale_price>1.5*purchase_price;
  ```

  如果不使用 UNION

  ```sql
  -- 参考答案:
  SELECT  product_id,product_name,product_type
         ,sale_price,purchase_price
    FROM product 
   WHERE sale_price < 800 
      OR sale_price > 1.5 * purchase_price;
  ```

- UNION 与 OR 谓词

  对于同一个表的两个不同的筛选结果集, 使用 UNION 对两个结果集取并集, 和把两个子查询的筛选条件用 OR 谓词连接, 会得到相同的结果, 但倘若要将两个不同的表中的结果合并在一起, 就不得不使用 UNION 了。

- UNION ALL 包含重复行

  SQL 语句的 UNION 会对多个查询的结果集进行合并和去重。只需要在 UNION 后面添加 ALL 关键字就可以实现在 UNION 的结果中保留重复行。

- bag 模型与 set 模型

  最主要的区别是：**bag 里面允许存在重复元素。**

  bag 模型的并运算：**1.该元素是否至少在一个 bag 里出现过, 2.该元素在两个 bag 中的最大出现次数**。 因此对于 A = {1,1,1,2,3,5,7}, B = {1,1,2,2,4,6,8} 两个 bag, 它们的并就等于 {1,1,1,2,2,3,4,5,6,7,8}。

- 隐式类型转换

  即使数据类型不完全相同, 也会通过隐式类型转换来将两个类型不同的列放在一列里显示, 例如字符串和数值类。

  ```sql
  SELECT SYSDATE(), SYSDATE(), SYSDATE()
   
   UNION
   
  SELECT 'chars', 123,  null
  ```

  说明时间日期类型和字符串,数值以及缺失值均能兼容。

### 4.1.3 MySQL 不支持交运算INTERSECT

虽然集合的交运算在SQL标准中已经出现多年了, 然而很遗憾的是, 截止到 MySQL 8.0 版本, MySQL 仍然不支持 INTERSECT 操作。

- bag 模型的交运算

  对于两个 bag, 他们的交运算会按照: **1.该元素是否同时属于两个 bag, 2.该元素在两个 bag 中的最小出现次数**这两个方面来进行计算. 因此对于 A = {1,1,1,2,3,5,7}, B = {1,1,2,2,4,6,8} 两个 bag, 它们的交运算结果就等于 {1,1,2}。

### 4.1.4 差集,补集与表的减法

- 差集

MySQL 8.0 还不支持 表的减法运算符 EXCEPT. 不过, 借助前边学过的NOT IN 谓词, 我们同样可以实现表的减法.

```sql
-- 使用 IN 子句的实现方法
SELECT * 
  FROM product
 WHERE product_id NOT IN (SELECT product_id 
                            FROM product2)
```

- EXCEPT ALL 与 bag 的差

  类似于UNION ALL, EXCEPT ALL 也是按出现次数进行减法, 也是使用bag模型进行运算.

  对于两个 bag, 他们的差运算会按照:**1.该元素是否属于作为被减数的 bag；2.该元素在两个 bag 中的出现次数这两个方面来进行计算。**

  只有属于被减数的bag的元素才参与EXCEP ALL运算, 并且差bag中的次数,等于该元素在两个bag的出现次数之差(差为零或负数则不出现). 因此对于 A = {1,1,1,2,3,5,7}, B = {1,1,2,2,4,6,8} 两个 bag, 它们的差就等于 {1,3,5,7}。

- INTERSECT 与 AND 谓词

  对于同一个表的两个查询结果而言, 他们的交INTERSECT实际上可以等价地将两个查询的检索条件用AND谓词连接来实现。

### 4.1.5 对称差

两个集合A,B的对称差是指那些仅属于A或仅属于B的元素构成的集合。

举例（**使用 NOT IN 实现两个表的差集**）

```sql
-- 使用 NOT IN 实现两个表的差集
SELECT * 
  FROM product
 WHERE product_id NOT IN (SELECT product_id FROM product2)
UNION
SELECT * 
  FROM product2
 WHERE product_id NOT IN (SELECT product_id FROM product)
```

## 4.2 连结（JOIN）

上一节的集合运算不能改变列的变化，虽然使用函数或者 CASE表达式等列运算, 可以增加列的数量, 但仍然只能从一张表中提供的基础信息列中获得一些"引申列", 本质上并不能提供更多的信息。

使用关联子查询也可以从其他表获取信息, 但**连结（JOIN）更适合从多张表获取信息**。

### 4.2.1 内连结（INNER JOIN）

- 使用内连结从两个表获取信息

内连结的语法格式是:

```sql
-- 内连结
FROM <tb_1> INNER JOIN <tb_2> ON <condition(s)>
```

关于内连结,需要注意以下三点:

1. 进行连结时需要在 FROM 子句中使用多张表.

2. 必须使用 ON 子句来指定连结条件.

3. SELECT 子句中的列最好按照 表名.列名 的格式来使用.

- 结合 WHERE 子句使用内连结

1. 第一种增加 WEHRE 子句的方式, 就是把上述查询作为子查询, 用括号封装起来, 然后在外层查询增加筛选条件。

2. 如果需要在使用内连结的时候同时使用 WHERE 子句对检索结果进行筛选, 则需要把 WHERE 子句写在 ON 子句的后边。

此时，查询的执行顺序为：**FROM 子句->WHERE 子句->SELECT 子句**。也就是说, 两张表是先按照连结列进行了连结, 得到了一张新表, 然后 WHERE 子句对这张新表的行按照两个条件进行了筛选, 最后, SELECT 子句选出了那些我们需要的列。

3. WHERE 子句中的条件直接添加在 ON 子句中, 这时候 ON 子句后最好用括号将连结条件和筛选条件括起来。
4. 由于连结多个表的操作很费时，在结合 WHERE 子句使用内连结的时候, 我们也可以更改任务顺序：采用任务分解的方法,先分别在两个表使用 WHERE 进行筛选,然后把上述两个子查询连结起来。

- 结合 GROUP BY 子句使用内连结

最简单的情形, 是在内连结之前就使用 GROUP BY 子句。

但是如果分组列和被聚合的列不在同一张表, 且二者都未被用于连结两张表, 则只能先连结, 再聚合。

```sql
SELECT SP.shop_id
      ,SP.shop_name
      ,MAX(P.sale_price) AS max_price
  FROMshopproduct AS SP
 INNER JOINproduct AS P
    ON SP.product_id = P.product_id
 GROUP BY SP.shop_id,SP.shop_name
```

- 自连结（SELF JOIN）

一张表也可以与自身作连结, 这种连接称之为自连结。

- 内连结与关联子查询

关联子查询可以使用内连结进行改写。

- 自然连结(NATURAL JOIN)

当两个表进行自然连结时, 会按照两个表中都包含的列名来进行等值内连结, 此时无需使用 ON 来指定连接条件。

```sql
SELECT *  FROM shopproduct NATURAL JOIN product
```

- 使用连结求交集

MySQL 8.0 里没有交集运算, 上面是通过并集和差集来实现求交集的。现在可以使用连结来实现求交集的运算。

练习题: 使用内连结求 product 表和 product2 表的交集.

```sql
SELECT P1.*
  FROM product AS P1
 INNER JOIN product2 AS P2
    ON (P1.product_id  = P2.product_id
   AND P1.product_name = P2.product_name
   AND P1.product_type = P2.product_type
   AND P1.sale_price   = P2.sale_price
   AND P1.regist_date  = P2.regist_date)
```

### 4.2.2 外连结(OUTER JOIN)

外连结会根据外连结的种类有选择地保留无法匹配到的行。

按照保留的行位于哪张表,外连结有三种形式: **左连结**, **右连结** 和 **全外连结**。

- 左连结会保存左表中无法按照 ON 子句匹配到的行, 此时对应右表的行均为缺失值; 
- 右连结则会保存右表中无法按照 ON 子句匹配到的行, 此时对应左表的行均为缺失值; 
- 全外连结则会同时保存两个表中无法按照 ON子句匹配到的行, 相应的另一张表中的行用缺失值填充.

三种外连结的对应语法分别为:

```sql
-- 左连结     
FROM <tb_1> LEFT  OUTER JOIN <tb_2> ON <condition(s)>
-- 右连结     
FROM <tb_1> RIGHT OUTER JOIN <tb_2> ON <condition(s)>
-- 全外连结
FROM <tb_1> FULL  OUTER JOIN <tb_2> ON <condition(s)>
```

#### 4.2.2.1 使用左连结

练习题: 统计每种商品分别在哪些商店有售, 需要包括那些在每个商店都没货的商品.

使用左连结的代码如下(注意区别于书上的右连结):

```sql
SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.sale_price
  FROM product AS P
  LEFT OUTER JOIN shopproduct AS SP
    ON SP.product_id = P.product_id;
```

- **外连结要点 1: 选取出单张表中全部的信息**
- **外连结要点 2: 使用 LEFT、RIGHT 来指定主表.**

#### 4.2.2.2 结合 WHERE 子句使用左连结

由于存在 null，使用 where 查询的时候需要额外注意。下面举例说明。

**练习题：**

使用外连结从 `shopproduct` 表和 `product` 表中找出那些在某个商店库存少于50的商品及对应的商店。

我们很自然会写出如下代码

```sql
SELECT P.product_id
       ,P.product_name
       ,P.sale_price
       ,SP.shop_id
       ,SP.shop_name
       ,SP.quantity
  FROM product AS P
  LEFT OUTER JOIN shopproduct AS SP
    ON SP.product_id = P.product_id
 WHERE quantity< 50
```

这样会导致找不到值为 null 的数据，因为 null 不支持常规的比较，如果强行比较，结果一定为 false。

可以试着把WHERE子句挪到外连结之前进行: 先写个子查询,然后再把这个子查询和主表连结起来。

```sql
SELECT P.product_id
      ,P.product_name
      ,P.sale_price
       ,SP.shop_id
      ,SP.shop_name
      ,SP.quantity 
  FROM product AS P
  LEFT OUTER JOIN-- 先筛选quantity<50的商品
   (SELECT *
      FROM shopproduct
     WHERE quantity < 50 ) AS SP
    ON SP.product_id = P.product_id
```

### 4.2.3 在 MySQL 中实现全外连结

全外连结本质上就是对左表和右表的所有行都予以保留, 能用 ON 关联到的就把左表和右表的内容在一行内显示, 不能被关联到的就分别显示, 然后把多余的列用缺失值填充.

遗憾的是, MySQL8.0 目前还不支持全外连结, 不过我们可以对左连结和右连结的结果进行 UNION 来实现全外连结.

## 4.3 多表连结

首先创建一个用于三表连结的表 Inventoryproduct。

建表语句如下:

```
CREATE TABLE Inventoryproduct
( inventory_id       CHAR(4) NOT NULL,
product_id         CHAR(4) NOT NULL,
inventory_quantity INTEGER NOT NULL,
PRIMARY KEY (inventory_id, product_id));
```

然后插入一些数据:

```sql
--- DML：插入数据
START TRANSACTION;
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0001', 0);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0002', 120);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0003', 200);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0004', 3);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0005', 0);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0006', 99);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0007', 999);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P001', '0008', 200);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0001', 10);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0002', 25);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0003', 34);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0004', 19);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0005', 99);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0006', 0);
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0007', 0 );
INSERT INTO Inventoryproduct (inventory_id, product_id, inventory_quantity)
VALUES ('P002', '0008', 18);
COMMIT;
```

### 4.3.1 多表进行内连结