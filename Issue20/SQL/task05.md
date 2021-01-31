# SQL 高级处理

## 5.1 窗口函数

窗口函数也称为OLAP函数。OLAP 是OnLine AnalyticalProcessing 的简称，意思是对数据库数据进行实时分析处理。

窗口函数的通用形式：

```sql
<窗口函数> OVER ([PARTITION BY <列名>]
                     ORDER BY <排序用列名>)  
```

主要的两个关键字：**PARTITON BY**和**ORDER BY**

- **PARTITON BY**是用来分组，即选择要看哪个窗口，类似于GROUP BY 子句的分组功能，但是PARTITION BY 子句并不具备GROUP BY 子句的汇总功能，并不会改变原始表中记录的行数。

- **ORDER BY**是用来排序，即决定窗口内，是按那种规则(字段)来排序的。可以通过关键字ASC/DESC来指定升序/降序。省略该关键字时会默认按照ASC，也就是升序进行排序。

- 举个例子:

  ```sql
  SELECT product_name
         ,product_type
         ,sale_price
         ,RANK() OVER (PARTITION BY product_type
                           ORDER BY sale_price) AS ranking
    FROM product  
  ```

## 5.2 窗口函数种类

窗口函数大体可以分为两类。

1. 一是 将SUM、MAX、MIN等聚合函数用在窗口函数中

2. 二是 RANK、DENSE_RANK等排序用的专用窗口函数

### 5.2.1 专用窗口函数

- **RANK函数**：排序
- **DENSE_RANK函数**：同样是计算排序，即使存在相同位次的记录，也不会跳过之后的位次。
- **ROW_NUMBER函数**：赋予唯一的连续位次。（就是不管相不相等，都是递增的。）

### 5.2.2 聚合函数在窗口函数上的使用

使用方法和上面的专用窗口函数一样，只是出来的结果是一个**累计**的聚合函数值。

例子：

```sql
SELECT  product_id
       ,product_name
       ,sale_price
       ,SUM(sale_price) OVER (ORDER BY product_id) AS current_sum
       ,AVG(sale_price) OVER (ORDER BY product_id) AS current_avg  
  FROM product;  
```

聚合函数结果是，按我们指定的排序（这里是product_id），**当前所在行及之前所有的行**的合计或均值。即累计到当前行的聚合。

## 5.3 窗口函数的的应用 - 计算移动平均

还可以指定更加详细的**汇总范围**。该汇总范围成为**框架(frame)。**

语法：

```sql
<窗口函数> OVER (ORDER BY <排序用列名>
                 ROWS n PRECEDING )  
                 
<窗口函数> OVER (ORDER BY <排序用列名>
                 ROWS BETWEEN n PRECEDING AND n FOLLOWING)
```

参数说明：

- PRECEDING（“之前”）， 将框架指定为 “截止到之前 n 行”，加上自身行

- FOLLOWING（“之后”）， 将框架指定为 “截止到之后 n 行”，加上自身行

- BETWEEN 1 PRECEDING AND 1 FOLLOWING，将框架指定为 “之前1行” + “之后1行” + “自身”

举例：

```sql
SELECT  product_id
       ,product_name
       ,sale_price
       ,AVG(sale_price) OVER (ORDER BY product_id
                               ROWS 2 PRECEDING) AS moving_avg
       ,AVG(sale_price) OVER (ORDER BY product_id
                               ROWS BETWEEN 1 PRECEDING 
                                        AND 1 FOLLOWING) AS moving_avg  
  FROM product  
```

**窗口函数注意事项：**

- 原则上，窗口函数只能在SELECT子句中使用。
- 窗口函数OVER 中的ORDER BY 子句并不会影响最终结果的排序。其只是用来决定窗口函数按何种顺序计算。

## 5.4 `GROUPING` 运算符

- ROLLUP - 计算合计及小计

常规的GROUP BY 只能得到每个分类的小计，有时候还需要计算分类的合计，可以用 ROLLUP关键字。

```sql
SELECT  product_type
       ,regist_date
       ,SUM(sale_price) AS sum_price
  FROM product
 GROUP BY product_type, regist_date WITH ROLLUP  
```

## 5.5 练习题

### 5.5.1 验证结果

请说出针对本章中使用的 product（商品）表执行如下 SELECT 语句所能得到的结果。

```sql
SELECT  product_id
       ,product_name
       ,sale_price
       ,MAX(sale_price) OVER (ORDER BY product_id) AS Current_max_price
  FROM product
```

首先会按照`product_id`对数据进行升序排序，然后，第 i 行的 `Current_max_price` 是指从第 1 行开始到第 i 行的 `sale_price` 最大值。

### 5.5.2 查询

继续使用product表，计算出按照登记日期（regist_date）升序进行排列的各日期的销售单价（sale_price）的总额。排序是需要将登记日期为NULL 的“运动 T 恤”记录排在第 1 位（也就是将其看作比其他日期都早）

```sql
SELECT  product_id, product_name, regist_date, sale_price, 
      SUM(sale_price) OVER (ORDER BY regist_date) AS total_price
FROM product  
```

我这里默认就是 `NULL FIRST`， 显示加上这个的话，反而查询不到结果。

答案中的第一种方法也很好：

```sql
-- ①regist_date为NULL时，显示“1年1月1日”。
SELECT regist_date, product_name, sale_price,
       SUM(sale_price) OVER (ORDER BY COALESCE(regist_date, CAST('0001-01-01' AS DATE))) AS current_sum_price
  FROM Product;
```

使用，`COALESCE(regist_date, CAST('0001-01-01' AS DATE)` 当 `regist_date` 为 `NULL` 时，自动替换成 `0001-01-01` 的日期格式。

### 5.5.3 思考题

思考题

① 窗口函数不指定PARTITION BY的效果是什么？（对所有数据进行排序。）

② 为什么说窗口函数只能在SELECT子句中使用？实际上，在ORDER BY 子句使用系统并不会报错。（参考下面的答案\~）

参考答案：

①：窗口函数不指定 `PARTITION BY` 就是针对排序列进行全局排序。
②：本质上是因为 `SQL` 语句的执行顺序：**`FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY`**。
如果在 `WHERE, GROUP BY, HAVING` 使用了窗口函数，就是说提前进行了一次排序，排序之后再去除记录、汇总、汇总过滤，第一次排序结果就是错误的，没有实际意义。而 `ORDER BY` 语句执行顺序在 `SELECT` 语句之后，自然是可以使用的。