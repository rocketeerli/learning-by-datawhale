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

## 练习四：连续出现的数字（难度：中等）

### Question

编写一个 SQL 查询，查找所有至少连续出现三次的数字。

```nohighlight
+----+-----+
| Id | Num |
+----+-----+
| 1  |  1  |
| 2  |  1  |
| 3  |  1  |
| 4  |  2  |
| 5  |  1  |
| 6  |  2  |
| 7  |  2  |
+----+-----+
```

例如，给定上面的 Logs 表， 1 是唯一连续出现至少三次的数字。

```nohighlight
+-----------------+
| ConsecutiveNums |
+-----------------+
| 1               |
+-----------------+
```

### Solution

1. 创建数据库表

```sql
CREATE TABLE `Logs`(
	`Id` INT NOT NULL AUTO_INCREMENT,
	`Num` INT NOT NULL,
	PRIMARY KEY (Id)
);
```

2. 插入数据

```sql
INSERT INTO `logs`(Num) VALUES(1), (1), (1), (2), (1), (2), (2);
```

3. 查询

```sql
SELECT logs1.Num AS ConsecutiveNums 
FROM `logs` logs1 
INNER JOIN `logs` logs2
INNER JOIN `logs` logs3
ON logs1.Id = logs2.Id - 1 AND 
	logs2.Id = logs3.Id - 1
WHERE	logs1.Num = logs2.Num AND 
	logs2.Num = logs3.Num;
```

使用内连结，先将三个相同的表连起来，再进行查询。

看了几个其他人的答案，基本都是三个相同的表进行连结的方法，就是连结条件写在哪里的区别了。我以为会有啥高级的方法。。。

## 练习五：树节点 （难度：中等）

### Question

对于**tree**表，*id*是树节点的标识，*p_id*是其父节点的*id*。

```nohighlight
+----+------+
| id | p_id |
+----+------+
| 1  | null |
| 2  | 1    |
| 3  | 1    |
| 4  | 2    |
| 5  | 2    |
+----+------+
```

每个节点都是以下三种类型中的一种：

- Root: 如果节点是根节点。
- Leaf: 如果节点是叶子节点。
- Inner: 如果节点既不是根节点也不是叶子节点。

写一条查询语句打印节点id及对应的节点类型。按照节点id排序。上面例子的对应结果为：

```nohighlight
+----+------+
| id | Type |
+----+------+
| 1  | Root |
| 2  | Inner|
| 3  | Leaf |
| 4  | Leaf |
| 5  | Leaf |
+----+------+
```

**说明**

- 节点’1’是根节点，因为它的父节点为NULL，有’2’和’3’两个子节点。
- 节点’2’是内部节点，因为它的父节点是’1’，有子节点’4’和’5’。
- 节点’3’，‘4’，'5’是叶子节点，因为它们有父节点但没有子节点。

下面是树的图形：

```
    1         
  /   \ 
 2    3    
/ \
4  5
```

**注意**

如果一个树只有一个节点，只需要输出根节点属性。

### Solution

1. 创建表：

```sql
SELECT id
FROM tree
CREATE TABLE tree (
	id INT NOT NULL AUTO_INCREMENT,
	p_id INT NULL,
	PRIMARY KEY(id)
);
```

2. 插入数据：

```sql
INSERT INTO tree(p_id) VALUE(NULL), (1), (1), (2), (2);
```

3. 查询：

```sql
SELECT t.id, case
when p_id IS NULL then 'Root'
when son_num IS NULL then 'Leaf'
else 'Inner' END AS `Type`
FROM tree t
LEFT JOIN (
	SELECT t2.id AS id, COUNT(*) AS son_num
		FROM tree t2
		JOIN tree t3
		ON t2.id = t3.p_id
		GROUP BY (t2.id)
) c
ON t.id = c.id
ORDER BY t.id;
```

有个简介的答案：

```sql
SELECT DISTINCT t.id, case
when t.p_id IS NULL then 'Root'
when c.id IS NULL then 'Leaf'
else 'Inner' END AS `Type`
FROM tree t
LEFT JOIN tree c
ON t.id = c.p_id
ORDER BY t.id;
```

## 练习六：至少有五名直接下属的经理 （难度：中等）

### Question

**Employee**表包含所有员工及其上级的信息。每位员工都有一个Id，并且还有一个对应主管的Id（ManagerId）。

```nohighlight
+------+----------+-----------+----------+
|Id    |Name 	  |Department |ManagerId |
+------+----------+-----------+----------+
|101   |John 	  |A 	      |null      |
|102   |Dan 	  |A 	      |101       |
|103   |James 	  |A 	      |101       |
|104   |Amy 	  |A 	      |101       |
|105   |Anne 	  |A 	      |101       |
|106   |Ron 	  |B 	      |101       |
+------+----------+-----------+----------+
```

针对**Employee**表，写一条SQL语句找出有5个下属的主管。对于上面的表，结果应输出：

```nohighlight
+-------+
| Name  |
+-------+
| John  |
+-------+
```

**注意:**

没有人向自己汇报。

### Solution

1. 创建表：

```sql
CREATE TABLE employee(
	`Id` INT NOT NULL,
	`Name` VARCHAR(25) NOT NULL,
	`Department` CHAR(1) NOT NULL,
	`ManagerId` INT NULL,
	PRIMARY KEY(Id)
);
```

2. 插入数据：

```sql
INSERT INTO employee VALUES
(101, "John", 'A', NULL),
(102, "Dan", 'A', 101),
(103, "James", 'A', 101),
(104, "Amy", 'A', 101),
(105, "Anne", 'A', 101),
(106, "Ron", 'B', 101);
```

3. 查询：

```sql
SELECT e1.`Name` AS `Name`
FROM employee e1
JOIN employee e2
ON e1.Id = e2.ManagerId AND e1.Id != e2.Id
GROUP BY e1.`Name`
HAVING COUNT(*) >= 5;
```

连结俩相同的表。。注意 `Having` 的使用。

这题，如果只选出主管，直接将 `ManagerId` 为 `NULL` 的挑出来，再做选择也成。

## 练习七: 分数排名 （难度：中等）

### Question

练习三的分数表，实现排名功能，但是排名需要是非连续的，如下：

```nohighlight
+-------+------+
| Score | Rank |
+-------+------+
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 3    |
| 3.65  | 4    |
| 3.65  | 4    |
| 3.50  | 6    |
+-------+------
```

### Solution

```sql
SELECT ROUND(`Score`, 2) AS `Score`,
RANK() over(ORDER BY `Score` DESC) AS `Rank`
FROM score
ORDER BY(`Score`) DESC;
```

注意 `RANK()` 窗口函数的使用，以及 `ORDER BY` 的 `DESC`。

还有，要注意两位小数点。。。(使用 `ROUND` 函数)

其实可以不使用最后的 `Order by`

```sql
SELECT ROUND(`Score`, 2) AS `Score`,
RANK() over(ORDER BY `Score` DESC) AS `Rank`
FROM score;
```

## 练习八：查询回答率最高的问题 （难度：中等）

### Question

求出**survey_log**表中回答率最高的问题，表格的字段有：**uid, action, question_id, answer_id, q_num, timestamp**。

uid是用户id；action的值为：“show”， “answer”， “skip”；当action是"answer"时，answer_id不为空，相反，当action是"show"和"skip"时为空（null）；q_num是问题的数字序号。

写一条sql语句找出回答率最高的问题。

**举例：**

**输入**

| uid  | action | question_id | answer_id | q_num | timestamp |
| :--- | :----- | :---------- | :-------- | :---- | :-------- |
| 5    | show   | 285         | null      | 1     | 123       |
| 5    | answer | 285         | 124124    | 1     | 124       |
| 5    | show   | 369         | null      | 2     | 125       |
| 5    | skip   | 369         | null      | 2     | 126       |

**输出**

| survey_log |
| :--------- |
| 285        |

**说明**

问题285的回答率为1/1，然而问题369的回答率是0/1，所以输出是285。

**注意：**最高回答率的意思是：同一个问题出现的次数中回答的比例。

### Solution

1. 创建数据库表：

```sql
CREATE TABLE survey_log (
	uid int NOT NULL,
	`action` VARCHAR(10) NOT NULL, 
	question_id INT NOT NULL, 
	answer_id INT NULL, 
	q_num INT NOT NULL, 
	`timestamp` INT NOT NULL 
);
```

2. 插入数据：

```sql
INSERT INTO survey_log VALUES
(5, "show", 285, NULL, 1, 123),
(5, "answer", 285, 124124,	1,	124),
(5, "show", 369, NULL, 2, 125),
(5, "skip", 369, NULL, 2, 126);
```

3. 查询：

- 使用子查询的方法：

```sql
SELECT question_id AS `survey_log`
FROM (
	SELECT question_id, MAX(a.answer_rate)
	FROM (
		SELECT question_id, 
			COUNT(answer_id) / COUNT(question_id) AS answer_rate
		FROM survey_log
		WHERE `action` != "show"
		GROUP BY (question_id)
	) a
) b;
```

- 使用排序的方法：

```sql
SELECT question_id AS `survey_log`
FROM (
	SELECT question_id, 
		COUNT(answer_id) / COUNT(question_id) AS answer_rate
	FROM survey_log
	WHERE `action` != "show"
	GROUP BY (question_id)
) a
ORDER BY a.answer_rate DESC
LIMIT 1;
```

## 练习九：各部门前3高工资的员工（难度：中等）

### Question

将项目7中的employee表清空，重新插入以下数据（其实是多插入5,6两行）：

```nohighlight
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
| 5  | Janet | 69000  | 1            |
| 6  | Randy | 85000  | 1            |
+----+-------+--------+--------------+
```

编写一个 SQL 查询，找出每个部门工资前三高的员工。例如，根据上述给定的表格，查询结果应返回：

```nohighlight
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| IT         | Randy    | 85000  |
| IT         | Joe      | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |
+------------+----------+--------+
```

此外，请考虑实现各部门前N高工资的员工功能。

### Solution

1. 建表：

```sql
CREATE TABLE Employee(
	`Id` INT NOT NULL AUTO_INCREMENT,
	`Name` VARCHAR(25) NOT NULL,
	`Salary` INT NOT NULL,
	`DepartmentId` INT NOT NULL,
	PRIMARY KEY (Id)
);
```

2. 插入数据：

```sql
INSERT INTO employee VALUES(1,'Joe',70000,1),
(2, 'Henry', 80000, 2),
(3, 'Sam', 60000, 2),
(4, 'Max', 90000, 1),
(5, 'Janet', 69000, 1),
(6, 'Randy', 85000, 1);
```

3. 查询

```sql
SELECT a.`Department`, a.`Employee`, a.Salary
FROM (SELECT d.`Name` AS `Department`, 
		e.`Name` AS `Employee`, Salary, 
		ROW_NUMBER() OVER (PARTITION BY d.Id
                  ORDER BY Salary DESC) AS ranking
FROM department d
INNER JOIN employee e
ON(d.Id = e.DepartmentId)) a
WHERE ranking <= 3;
```

首先使用窗口函数进行升序排序，然后根据排序的序号进行筛选。