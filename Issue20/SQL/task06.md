# Task06 综合练习

## 练习一: 各部门工资最高的员工（难度：中等）

### Question

创建Employee 表，包含所有员工信息，每个员工有其对应的 Id, salary 和 department Id。

```nohighlight
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
```

创建Department 表，包含公司所有部门的信息。

```nohighlight
+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+
```

编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。

```nohighlight
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+
```

### Solution

1. 创建 Employee 表：

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

2. 创建 Department 表：

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

3. 编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。

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

### Question

小美是一所中学的信息科技老师，她有一张 seat 座位表，平时用来储存学生名字和与他们相对应的座位 id。

其中纵列的**id**是连续递增的

小美想改变相邻俩学生的座位。

你能不能帮她写一个 SQL query 来输出小美想要的结果呢？

请创建如下所示seat表：

**示例：**

```nohighlight
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Abbot   |
|    2    | Doris   |
|    3    | Emerson |
|    4    | Green   |
|    5    | Jeames  |
+---------+---------+
```

假如数据输入的是上表，则输出结果如下：

```nohighlight
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Doris   |
|    2    | Abbot   |
|    3    | Green   |
|    4    | Emerson |
|    5    | Jeames  |
+---------+---------+
```

**注意：**
如果学生人数是奇数，则不需要改变最后一个同学的座位。

### Solution

1. 创建数据库：

```sql
CREATE TABLE seat (
	id INT UNSIGNED NOT NULL AUTO_INCREMENT,
	student VARCHAR(25) NOT NULL,
	PRIMARY KEY (id)
);
```

2. 插入数据：

```sql
INSERT INTO seat (student) VALUES('Abbot'),
('Doris'), ("Emerson"), ('Green'), ('Jeames');
```

3. 查询：

```sql
SELECT s1.id, s2.student
FROM seat s1
LEFT JOIN seat s2
ON s1.id = (SELECT MAX(id) FROM seat) AND s1.id % 2 = 1 AND s1.id = s2.id
OR s1.id % 2 = 1 AND s2.id = s1.id + 1
OR s1.id % 2 = 0 AND s2.id = s1.id - 1;
```

也可以通过使用 case 来更改 id 来达到同样的效果。

```sql
SELECT case
when MOD(id, 2) = 1 AND id = largest then id
when MOD(id, 2) = 1 then id+1
else id-1 END AS id, student
FROM seat, (SELECT MAX(id) AS largest FROM seat) AS a
ORDER BY id;
```

## 练习三: 分数排名（难度：中等）

### Question

编写一个 SQL 查询来实现分数排名。如果两个分数相同，则两个分数排名（Rank）相同。请注意，平分后的下一个名次应该是下一个连续的整数值。换句话说，名次之间不应该有“间隔”。

创建以下score表：

```nohighlight
+----+-------+
| Id | Score |
+----+-------+
| 1  | 3.50  |
| 2  | 3.65  |
| 3  | 4.00  |
| 4  | 3.85  |
| 5  | 4.00  |
| 6  | 3.65  |
+----+-------+
```

例如，根据上述给定的 Scores 表，你的查询应该返回（按分数从高到低排列）：

```nohighlight
+-------+------+
| Score | Rank |
+-------+------+
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |
+-------+------+
```

### Solution

1. 创建数据库：

```sql
CREATE TABLE score(
	Id INT NOT NULL AUTO_INCREMENT,
	Score DOUBLE NOT NULL,
	PRIMARY KEY(Id)
);
```

2. 插入数据：

```sql
INSERT INTO score(Score) VALUES 
(3.50), (3.65), (4.00), (3.85), (4.00), (3.65);
```

3. 排序：

```sql
SELECT 
	`Score`, 
	DENSE_RANK() over (ORDER BY `Score` DESC) `Rank`
FROM score;
```

