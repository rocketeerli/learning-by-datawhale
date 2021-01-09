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