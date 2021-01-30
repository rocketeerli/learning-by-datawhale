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