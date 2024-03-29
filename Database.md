## LeetCode Problem - Database

[toc]

### [175. Combine Two Tables](https://leetcode.com/problems/combine-two-tables/)

Write an SQL query to report the first name, last name, city, and state of each person in the Person table. If the address of a personId is not present in the Address table, report `null` instead.

```sql
select p.firstName, p.lastName, ad.city, ad.state 
from person as p
left join address as ad
on p.personId = ad.personId;
```

### [176. Second Highest Salary](https://leetcode.com/problems/second-highest-salary/)

Write an SQL query to report the second highest salary from the `Employee` table. If there is no second highest salary, the query should report `null`.

```sql
select
(select distinct salary
from employee
order by salary DESC
limit 1,1) as SecondHighestSalary;
```

### [177. Nth Highest Salary](https://leetcode.com/problems/nth-highest-salary/)

Write an SQL query to report the `nth` highest salary from the `Employee` table. If there is no `nth` highest salary, the query should report `null`.

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    SET n = n-1;
    RETURN (
        # Write your MySQL query statement below.
        select distinct salary
        from employee
        order by salary DESC
        limit 1 offset n
    );
END
```

### [178. Rank Scores](https://leetcode.com/problems/rank-scores/)

Write an SQL query to rank the scores. The ranking should be calculated according to the following rules:

The scores should be ranked from the highest to the lowest.

If there is a tie between two scores, both should have the same ranking.

After a tie, the next ranking number should be the next consecutive integer value. In other words, there should be no holes between ranks.

Return the result table ordered by score in descending order.

Using dense_rank():

```sql
select 
  score, dense_rank() over (order by score DESC) as 'rank'
from Scores;
```

Another solu:

```sql
SELECT S.Score, COUNT(S2.Score) AS Rank FROM Scores S,
(SELECT DISTINCT Score FROM Scores) S2
WHERE S.Score<=S2.Score
GROUP BY S.Id 
ORDER BY S.Score DESC;
```

### [180. Consecutive Numbers](https://leetcode.com/problems/consecutive-numbers/)

Write an SQL query to find all numbers that appear at least three times consecutively.

Return the result table in **any order**.

**Explain**: As we're traversing the rows, we're trying to find the ideal condition for the 1st num we encounter, that is the two consecutive ids having the same num.

```sql
select distinct num as ConsecutiveNums
from logs
where (id + 1, num) in (select id,num from logs) and (id + 2, num) in (select id,num from logs);
```

### [181. Employees Earning More Than Their Managers](https://leetcode.com/problems/employees-earning-more-than-their-managers/)

Write an SQL query to find the employees who earn more than their managers.

Return the result table in **any order**.

```sql
select subresult.name as Employee
from (select a.name, a.salary as sa, b.salary as sb
      from employee a 
      left join employee b
      on a.managerId = b.id) subresult
where subresult.sa > subresult.sb;
```

### [182. Duplicate Emails](https://leetcode.com/problems/employees-earning-more-than-their-managers/)

Write an SQL query to report all the duplicate emails. Note that it's guaranteed that the email field is not NULL.

Return the result table in **any order**.

```sql
select email
from person
group by email
having count(email)>1;
```

### [183. Customers Who Never Order](https://leetcode.com/problems/customers-who-never-order/description/)

Write an SQL query to report all customers who never order anything.

Return the result table in **any order**.

```sql
select c.name as Customers
from customers c
left join orders o
on c.id = o.customerid
where o.id is null;
```

### [184. Department Highest Salary](https://leetcode.com/problems/department-highest-salary/)

Write an SQL query to find employees who have the highest salary in each of the departments.

Return the result table in **any order**.

```sql
select result.Department, result.Employee, result.Salary
from (select d.name as Department, e.name as Employee, e.salary as Salary,
      dense_rank() over (partition by e.departmentId order by e.salary DESC) d_rank
      from employee e
      left join department d
      on e.departmentId = d.id) result
where result.d_rank = 1;
```

### [185. Department Top Three Salaries](https://leetcode.com/problems/department-top-three-salaries/)

A company's executives are interested in seeing who earns the most money in each of the company's departments. A **high earner** in a department is an employee who has a salary in the **top three unique** salaries for that department.

Write an SQL query to find the employees who are **high earners** in each of the departments.

Return the result table **in any order**.

```sql
select result.Department, result.Employee, result.Salary
from (select d.name as Department, e.name as Employee, e.salary as Salary,
      dense_rank() over (partition by e.departmentId order by e.salary DESC) d_rank
      from employee e
      left join department d
      on e.departmentId = d.id) result
where result.d_rank < 4;
```

### [196. Delete Duplicate Emails](https://leetcode.com/problems/delete-duplicate-emails/)

Write an SQL query to **delete** all the duplicate emails, keeping only one unique email with the smallest `id`. Note that you are supposed to write a `DELETE` statement and not a `SELECT` one.

After running your script, the answer shown is the `Person` table. The driver will first compile and run your piece of code and then show the `Person` table. The final order of the `Person` table **does not matter**.

```sql
delete from person
where id not in 
  (select a.id from
    (select min(id) id
    from person
    group by email) a);
```

```sql
delete p1 
from Person p1, Person p2
where p1.email=p2.email and p1.Id > p2.Id;
```

### [197. Rising Temperature](https://leetcode.com/problems/rising-temperature/)

Write an SQL query to find all dates' `Id` with higher temperatures compared to its previous dates (yesterday).

Return the result table in **any order**.

> DATEDIFF(expr1,expr2)
>
> DATEDIFF() returns expr1 − expr2 expressed as a value in days from one date to the other. expr1 and expr2 are date or date-and-time expressions. Only the date parts of the values are used in the calculation.

```sql
select a.id
from
  (select id, recordDate, temperature,
  lead(temperature, 1) over (order by recordDate DESC) prevt,
  lead(recordDate, 1) over (order by recordDate DESC) prevd
  from weather) a
where a.prevt < a.temperature and datediff(a.recordDate, a.prevd) = 1;
```

### [262. Trips and Users](https://leetcode.com/problems/trips-and-users/description/)

The **cancellation rate** is computed by dividing the number of canceled (by client or driver) requests with unbanned users by the total number of requests with unbanned users on that day.

Write a SQL query to find the **cancellation rate** of requests with unbanned users (**both client and driver must not be banned**) each day between `"2013-10-01"` and `"2013-10-03"`. Round `Cancellation Rate` to **two decimal** points.

Return the result table in **any order**.

```sql
select b.request_at as Day,
round(count(case when b.status<>'completed' then b.id end) / count(b.id), 2)  'Cancellation Rate'
from
  (select a.id, a.request_at, a.status, a.driver_id, a.c_ban, u.banned as d_ban
  from
    (select t.id, t.client_id, t.driver_id, u.banned as c_ban, t.request_at, t.status
    from trips t
    left join users u
    on t.client_id = u.users_id) a
  left join users u
  on a.driver_id = u.users_id) b
where b.c_ban = 'No' and b.d_ban = 'No' and b.request_at between '2013-10-01' and '2013-10-03'
group by b.request_at;
```

### [511. Game Play Analysis I](https://leetcode.com/problems/game-play-analysis-i/)

Write an SQL query to report the **first login date** for each player.

Return the result table in **any order**.

```sql
select player_id, min(event_date) first_login
from activity
group by player_id;
```

### [512. Game Play Analysis II](https://leetcode.com/problems/game-play-analysis-ii/)

```sql
select a.player_id, a.device_id first_login
from
  (select player_id, device_id, event_date,
  dense_rank() over (partition by player_id order by event_date) d_rank
  from activity) a
where a.d_rank = 1;
```

### [534. Game Play Analysis III](https://leetcode.com/problems/game-play-analysis-iii/)

```sql
select player_id, event_date, 
sum(games_played) over (partition by player_id order by event_date) gmaes_played_so_far
from activity;
```

### [550. Game Play Analysis IV](https://leetcode.com/problems/game-play-analysis-iv/)

Write an SQL query to report the **fraction** of players that logged in again on the day after the day they first logged in, **rounded to 2 decimal places**. 

In other words, you need to count the number of players that logged in for at least two consecutive days starting from their first login date, then divide that number by the total number of players.

```sql
select 
  round(count(distinct case when t1.rank_date=1 and t1.date_diff=1 then t1.player_id end) / count(distinct t1.player_id), 2) fraction
from
  (select player_id,
    dense_rank() over (partition by player_id order by event_date) rank_date,
    datediff(lead(event_date) over (partition by player_id order by event_date), event_date) date_diff
  from activity) t1
;
```

### [569. Median Employee Salary](https://leetcode.com/problems/median-employee-salary/)

```sql

```

### [595. Big Countries](https://leetcode.com/problems/big-countries/)

Write an SQL query to report the **first login date** for each player.

Return the result table in **any order**.

- 使用`where OR`

  ```
  select name, population, area 
  from World 
  where area > 3000000 or population > 25000000;
  ```

- 使用`UNION`

  ```
  select name, population, area 
  from World 
  where area > 3000000
  union
  select name, population, area
  from World
  where population > 25000000
  ;
  ```

补充说明：Solution中解释道，使用`UNION`会比`OR`快上一丢丢。

> Suppose we are searching population and area, Given that MySQL usually uses one one index per table in a given query, so when it uses the 1st index rather than 2nd index, it would still have to do a table-scan to find rows that fit the 2nd index.

因为前者查询的时候只用到第一个索引，对第二个条件查询时，也就是`population`，使用的是全表扫描，于是浪费了一些时间。但是想想背后的代价，想必是以空间来换时间

### [596. Classes More Than 5 Students](https://leetcode.com/problems/classes-more-than-5-students/)



```sql
select class
from courses
group by class
having count(distinct student) >= 5
;
```

### [620. Not Boring Movies](https://leetcode.com/problems/not-boring-movies/)



```sql
select *
from cinema
where mod(id, 2) = 1 and description != 'boring'
order by rating DESC
;
```

### [626. Exchange Seats](https://leetcode.com/problems/exchange-seats/)



开始理解错了，以为要update。

- `CASE`

  ```
  SELECT ( CASE
             WHEN MOD(id, 2) != 0
                  AND counts != id THEN id + 1
             WHEN MOD(id, 2) != 0
                  AND counts = id THEN id
             ELSE id - 1
           end ) AS id,
         student
  FROM   seat,
         (SELECT Count(*) AS counts
          FROM   seat) AS seat_counts
  ORDER  BY id ASC;;
  ```

- `COALESCE()`

  - 第一步：使用`XOR`，但是不能直接使用排序因为id为5的被换成了6。

    ```
    SELECT id, (id+1)^1-1, student FROM seat;
    ```

    ```
    +------+------+----------+------------+---------+
    | id   | id+1 | (id+1)^1 | (id+1)^1-1 | student |
    +------+------+----------+------------+---------+
    |    1 |    2 |        3 |          2 | Abbot   |
    |    2 |    3 |        2 |          1 | Doris   |
    |    3 |    4 |        5 |          4 | Emerson |
    |    4 |    5 |        4 |          3 | Green   |
    |    5 |    6 |        7 |          6 | Jeame   |
    +------+------+----------+------------+---------+
    ```

    - 第二步：使用`LEFT JOIN`链接。

      ```
      SELECT *
      FROM   seat s1
             LEFT JOIN seat s2
                    ON ( s1.id + 1 )^1 - 1 = s2.id
      ORDER  BY s1.id;
      ```

      ```
      +------+---------+------+---------+
      | id   | student | id   | student |
      +------+---------+------+---------+
      |    1 | Abbot   |    2 | Doris   |
      |    2 | Doris   |    1 | Abbot   |
      |    3 | Emerson |    4 | Green   |
      |    4 | Green   |    3 | Emerson |
      |    5 | Jeame   | NULL | NULL    |
      +------+---------+------+---------+
      ```

  - 第三步：使用`COALESCE()`

    ```
    SELECT s1.id,
           Coalesce(s2.student, s1.student) as student
    FROM   seat s1
           LEFT JOIN seat s2
                  ON ( s1.id + 1 )^1 - 1 = s2.id
    ORDER  BY s1.id
    ;
    ```

参考：Mysql文档

> Returns the first non-NULL value in the list, or NULL if there are no non-NULL values.
> The return type of COALESCE() is the aggregated type of the argument types.

### [627. Swap Salary](https://leetcode.com/problems/swap-salary/)

Write an SQL query to swap all `'f'` and `'m'` values (i.e., change all `'f'` values to `'m'` and vice versa) with a **single update statement** and no intermediate temporary tables.

Note that you must write a single update statement, **do not** write any select statement for this problem.

- use `case`

  ```sql
  update salary
  SET sex = 
   	case when sex='m' then 'f' else 'm' end;
  ```

- use `if`

  ```sql
  UPDATE salary
  SET sex = 
  	IF(sex='m', 'f', 'm');
  ```

- use `XOR`

  ```sql
  update salary
  set sex = 
  	CHAR(ASCII('f') ^ ASCII('m') ^ ASCII(sex));
  ```

### [title](link)

text

```sql

```

