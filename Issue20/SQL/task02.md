# 2. 基础查询与排序

## 2.1 `SELECT` 语句基础

- 从表中选数据

基本SELECT语句包含了SELECT和FROM两个子句（clause）。示例如下：

```sql
SELECT <列名>, 
  FROM <表名>;
```

- 选出符合条件的数据

SELECT 语句通过WHERE子句来指定查询数据的条件。

```sql
SELECT <列名>, ……
  FROM <表名>
 WHERE <条件表达式>;
```

- 相关规则
  - 星号（*）代表全部列的意思。
  - SQL中**可以随意使用换行符**，不影响语句执行（**但不可插入空行**）。
  - 设定汉语别名时需要使用双引号（"）括起来。
  - 在SELECT语句中使用 `DISTINCT` 可以删除重复行。
  - 注释是 SQL 语句中用来标识说明或者注意事项的部分。分为1行注释"-- “和多行注释两种”/* */"。

## 2.2 算数运算符和比较运算符

- 算数运算符：+、-、\*、/（加减乘除）
- 比较运算符：=、<>、>、<、>=、<=，其中 `<>` 表示不相等
- 相关规则
  - 字符串类型的数据原则上按照字典顺序进行排序，不能与数字的大小顺序混淆。
  - 希望选取NULL记录时，需要在条件表达式中使用IS NULL运算符。希望选取不是NULL的记录时，需要在条件表达式中使用IS NOT NULL运算符。

## 2.3 逻辑运算符

- `NOT` 运算符

想要表示“不是……”时，除了前文的<>运算符外，还存在另外一个表示否定、使用范围更广的运算符：NOT。

注意：**NOT不能单独使用**。

- `AND` 运算符 和 `OR` 运算符

**AND 运算符优先于 OR 运算符**。

- NULL的真值

NULL 的结果既不为真，也不为假，因为并不知道这样一个值。这时真值是除真假之外的第三种值——**不确定**（UNKNOWN）。

一般的逻辑运算并不存在这第三种值。`SQL` 之外的语言也基本上只使用真和假这两种真值。与通常的逻辑运算被称为二值逻辑相对，只有 `SQL` 中的逻辑运算被称为三值逻辑。

## 练习题

### 2.1 编写一条SQL语句，从 `product`（商品）表中选取出“登记日期（ `regist` 在2009年4月28日之后”的商品，查询结果要包含 `product_name` 和 `regist_date` 两列。

```sql
SELECT product_name, regist_date 
  FROM product
 WHERE regist_date >= "2009-04-28";
```

### 2.2 请说出对 `product` 表执行如下3条 `SELECT` 语句时的返回结果。

**任何值与NULL值进行比较运算的真值结果都是UNKNOWN。**

1. ```sql
   SELECT *
     FROM product
    WHERE purchase_price = NULL;
   ```

   无数据。因为选取 `NULL` 记录时，需要在条件表达式中使用 `IS NULL` 运算符。

2. ```sql
   SELECT *
     FROM product
    WHERE purchase_price <> NULL;
   ```

   无数据。因为选取不是 `NULL` 的记录时，需要在条件表达式中使用 `IS NOT NULL` 运算符。

3. ```sql
   SELECT *
     FROM product
    WHERE product_name > NULL;
   ```

   无数据。

### 2.3 代码清单2-22（2-2节）中的SELECT语句能够从product表中取出“销售单价（saleprice）比进货单价（purchase price）高出500日元以上”的商品。请写出两条可以得到相同结果的SELECT语句。执行结果如下所示。

```sql
product_name | sale_price | purchase_price 
-------------+------------+------------
T恤衫         |   1000    | 500
运动T恤       |    4000   | 2800
高压锅        |    6800   | 5000
```

1. ```sql
   SELECT product_name, sale_price, purchase_price
   FROM product
   WHERE sale_price - purchase_price  >= 500;
   ```

2. ```sql
   SELECT product_name, sale_price, purchase_price
   FROM product
   WHERE NOT sale_price < 500 + purchase_price;
   ```

### 2.4 请写出一条SELECT语句，从product表中选取出满足“销售单价打九折之后利润高于100日元的办公用品和厨房用具”条件的记录。查询结果要包括product_name列、product_type列以及销售单价打九折之后的利润（别名设定为profit）。

提示：销售单价打九折，可以通过saleprice列的值乘以0.9获得，利润可以通过该值减去purchase_price列的值获得。

```sql
SELECT product_name, product_type, sale_price * 0.9 - purchase_price as profit
FROM product
WHERE sale_price * 0.9 - purchase_price > 100 AND 
	(product_type = "办公用品" OR product_type = "厨房用具");
```

## 2.4 对表进行聚合查询（`COUNT`、`SUM`、`AVG`、`MAX`、`MIN`）

- 聚合函数：SQL中用于汇总的函数叫做聚合函数。`COUNT`、`SUM`、`AVG`、`MAX`、`MIN` （`MAX`和`MIN`也可用于非数值型数据）

**注意：**

1. COUNT函数的结果根据参数的不同而不同。COUNT(\*)会得到包含NULL的数据行数，而COUNT(<列名>)会得到NULL之外的数据行数。
2. MAX/MIN函数几乎适用于所有数据类型的列。SUM/AVG函数只适用于数值类型的列。
3. 想要计算值的种类或者删除重复数据时，可以在COUNT函数的参数中使用DISTINCT。`SELECT COUNT(DISTINCT product_type) FROM product;`

## 2.5 对表进行分组（`GROUP BY`)

- 定义：

之前使用聚合函数都是会整个表的数据进行处理，当你想将进行分组汇总时（即：将现有的数据按照某列来汇总统计），可以使用 `GROUP BY`

- 语法：

```sql
SELECT <列名1>,<列名2>, <列名3>, ……
  FROM <表名>
 GROUP BY <列名1>, <列名2>, <列名3>, ……;
```

- 扩展：

为什么我的 MySQL8.0 环境运行不使用 GROUP BY 子句的代码时会报错 ？
这是因为 MYSQL8.0 默认开启了 `ONLY_FULL_GROUP_BY` 查询模式，这种模式下不允许 SELECT list 中出现语义不明的列，用来保证 SQL 语句 “分组聚合” 的合法性。这种模式采用了与Oracle、DB2等数据库相同的处理方式。简单来说就是：**对于用到 GROUP BY 的 SELECT 语句，查询出来的列必须是 GROUP BY 后面声明的列，或者是聚合函数里面的列。**

查询当前客户端 `sql_mode` 的代码如下：

```sql
SELECT @@SESSION.sql_mode;
```

GROUP BY 子句就像切蛋糕那样将表进行了分组。

**注意：**

1. 在 GROUP BY 子句中指定的列称为**聚合键**或者**分组列**。
2. 聚合键中包含 NULL 时，会将 NULL 作为一组特殊数据进行处理。
3. `GROUP BY` 的子句书写顺序有严格要求。```1 SELECT → 2. FROM → 3. WHERE → 4. GROUP BY```
4. SELECT子句中可以通过AS来指定别名，但在GROUP BY中不能使用别名。(在DBMS中 ,SELECT子句在GROUP BY子句后执行。)

## 2.6 为聚合结果指定条件（`HAVING`）

### `HAVING` 得到特地分组。

- 问题：

将表使用GROUP BY分组后，怎样才能只取出其中特定几组？

这里WHERE不可行，因为，WHERE子句只能指定记录（行）的条件，而不能用来指定组的条件（例如，“数据行数为 2 行”或者“平均值为 500”等）。

- 答案：

可以在GROUP BY后使用HAVING子句。

HAVING的用法类似WHERE。

### `HAVING` 特点

HAVING子句用于对分组进行过滤，可以使用数字、聚合函数和GROUP BY中指定的列名（聚合键）。

- 使用数字

```sql
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type
HAVING COUNT(*) = 2;
```

- 错误形式（因为product_name不包含在GROUP BY聚合键中）

```sql
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type
HAVING product_name = '圆珠笔';
```

## 2.7 对查询结果进行排序（`ORDER BY`）

- 定义

SQL中的执行结果是随机排列的，当需要按照特定顺序排序时，可已使用**ORDER BY**子句。

- 语法

```sql
SELECT <列名1>, <列名2>, <列名3>, ……
  FROM <表名>
 ORDER BY <排序基准列1>, <排序基准列2>, ……
```

默认为升序排列，降序排列为`DESC`

```sql
-- 降序排列
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY sale_price DESC;
-- 多个排序键
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY sale_price, product_id;
-- 当用于排序的列名中含有NULL时，NULL会在开头或末尾进行汇总。
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY purchase_price;
```

