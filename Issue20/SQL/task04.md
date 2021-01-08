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

- 

