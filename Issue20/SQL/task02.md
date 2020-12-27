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

