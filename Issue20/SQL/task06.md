# Task06 综合练习

## 练习一: 各部门工资最高的员工（难度：中等）

创建 Employee 表：

```sql
CREATE TABLE Employee(
	`Id` INT NOT NULL,
	`Name` VARCHAR(25) NOT NULL,
	`Salary` INT,
	`DepartmentId` INT,
	PRIMARY KEY (Id)
);
```

插入数据：

```sql
INSERT INTO employee VALUES(1,'Joe',70000,1),
(2, 'Henry', 80000, 2),
(3, 'Sam', 60000, 2),
(4, 'Max', 90000, 1);
```

创建 Department 表：

```sql
CREATE TABLE Department(
	`Id` INT NOT NULL,
	`Name` VARCHAR(25) NOT NULL,
	PRIMARY KEY(Id)
);
```

插入数据：

```sql
INSERT INTO department VALUES(1, 'IT');
INSERT INTO department VALUES(2, "Sales");
```

编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。

```sql
SELECT 
	d.`Name` AS `Department`,
	e.`Name` AS `Employee`,
	`Salary`
FROM
employee e INNER JOIN department d
ON e.DepartmentId = d.Id
WHERE e.Salary IN (
	SELECT MAX(Salary) 
	FROM employee 
	GROUP BY DepartmentId
)
ORDER BY Salary DESC;
```

网上的一些答案是错的，不能直接在 `e.Name` 外加上 max 函数。

为了得到与示例答案一样的结果顺序，这里加了 `order by`。

## 练习二: 换座位（难度：中等）