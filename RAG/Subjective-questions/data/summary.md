# baseline
| id | total | get | detail | question1 | question2 | question3 | question4 | question5 |
|----|----|----|----|----|----|----|----|----|
| 1 | 34 | 4.5 | 0/7 2/8 1/12 1.5/7 | 0 | 2 | 1 | 1.5 | None |
| 2 | None | None | 此问不参与智能打分 | None | None | None | None | None |
| 3 | 30 | 9 | 4/13 2/5 2/7 1/5 | 4 | 2 | 2 | 1 | None |
| 4 | 34 | 1.5 | None | None | None | None | None | None |
| 5 | 24 | 3.5 | None | None | None | None | None | None |
| 6 | 31 | 6.5 | 0/9 2/6 4.5/16 | 0 | 2 | 4.5 | None | None |
| 7 | 20 | 4 | None | None | None | None | None | None |
| 8 | 30 | 4 | 1/7 1/7 1/7 1/7 0/2 | 1 | 1 | 1 | 1 | 0 |
| 9 | 23 | 4 | None | None | None | None | None | None |
| 10 | 22 | 9 | None | None | None | None | None | None |

# rag
| id | total | get | detail | question1 | question2 | question3 | question4 | question5 |
|----|----|----|----|----|----|----|----|----|
| 1 | 34 | 6 | 0/7 3/8 1.5/12 1.5/7 | 0 | 3 | 1.5 | 1.5 | None |
| 2 | None | None | none | None | None | None | None | None |
| 3 | 30 | 8 | 3/13 2/5 2/7 1/5 | 3 | 2 | 2 | 1 | None |
| 4 | 34 | 2.5 | None | None | None | None | None | None |
| 5 | 24 | 2.5 | None | None | None | None | None | None |
| 6 | 31 | 4.5 | 0/9 1/6 3.5/16 | 0 | 1 | 3.5 | None | None |
| 7 | 20 | 4 | None | None | None | None | None | None |
| 8 | 30 | 7.5 | 1/7 1.5/7 2/7 1/2 | 1 | 1.5 | 2 | 1 | None |
| 9 | 23 | 7 | None | None | None | None | None | None |
| 10 | 22 | 11 | None | None | None | None | None | None |

# mvrag
| id | total | get | detail | question1 | question2 | question3 | question4 | question5 |
|----|----|----|----|----|----|----|----|----|
| 1 | 34 | 5 | 0.5/7 0/8 3/12 1.5/7 | 0.5 | 0 | 3 | 1.5 | None |
| 2 | None | None | none | None | None | None | None | None |
| 3 | 30 | 10.5 | 3/13 3/5 3/7 1.5/5 | 3 | 3 | 3 | 1.5 | None |
| 4 | 34 | 3.5 | None | None | None | None | None | None |
| 5 | 24 | 2.5 | None | None | None | None | None | None |
| 6 | 31 | 8.5 | 2/9 2/6 4.5/16 | 2 | 2 | 4.5 | None | None |
| 7 | 20 | 4 | None | None | None | None | None | None |
| 8 | 30 | 5.5 | 2/7 1.5/7 0/7 2/7 0/2 | 2 | 1.5 | 0 | 2 | 0 |
| 9 | 23 | 5 | None | None | None | None | None | None |
| 10 | 22 | 9 | None | None | None | None | None | None |

# Summary Table

| id  | total_baseline | get_baseline | detail_baseline                | q1_baseline | q2_baseline | q3_baseline | q4_baseline | q5_baseline | total_rag | get_rag | detail_rag                     | q1_rag | q2_rag | q3_rag | q4_rag | q5_rag | total_mvrag | get_mvrag | detail_mvrag                  | q1_mvrag | q2_mvrag | q3_mvrag | q4_mvrag | q5_mvrag |
|-----|----------------|--------------|--------------------------------|-------------|-------------|-------------|-------------|-------------|-----------|---------|--------------------------------|--------|--------|--------|--------|--------|-------------|-----------|--------------------------------|----------|----------|----------|----------|----------|
| 1   | 34             | 4.5          | 0/7 2/8 1/12 1.5/7             | 0           | 2           | 1           | 1.5         | None        | 34        | 6       | 0/7 3/8 1.5/12 1.5/7           | 0      | 3      | 1.5    | 1.5    | None   | 34          | 5         | 0.5/7 0/8 3/12 1.5/7           | 0.5      | 0        | 3        | 1.5      | None     |
| 2   | None           | None         | 此问不参与智能打分             | None        | None        | None        | None        | None        | None      | None    | none                           | None   | None   | None   | None   | None   | None        | None      | none                           | None     | None     | None     | None     | None     |
| 3   | 30             | 9            | 4/13 2/5 2/7 1/5               | 4           | 2           | 2           | 1           | None        | 30        | 8       | 3/13 2/5 2/7 1/5               | 3      | 2      | 2      | 1      | None   | 30          | 10.5      | 3/13 3/5 3/7 1.5/5             | 3        | 3        | 3        | 1.5      | None     |
| 4   | 34             | 1.5          | None                           | None        | None        | None        | None        | None        | 34        | 2.5     | None                           | None   | None   | None   | None   | None   | 34          | 3.5       | None                           | None     | None     | None     | None     | None     |
| 5   | 24             | 3.5          | None                           | None        | None        | None        | None        | None        | 24        | 2.5     | None                           | None   | None   | None   | None   | None   | 24          | 2.5       | None                           | None     | None     | None     | None     | None     |
| 6   | 31             | 6.5          | 0/9 2/6 4.5/16                 | 0           | 2           | 4.5         | None        | None        | 31        | 4.5     | 0/9 1/6 3.5/16                 | 0      | 1      | 3.5    | None   | None   | 31          | 8.5       | 2/9 2/6 4.5/16                 | 2        | 2        | 4.5      | None     | None     |
| 7   | 20             | 4            | None                           | None        | None        | None        | None        | None        | 20        | 4       | None                           | None   | None   | None   | None   | None   | 20          | 4         | None                           | None     | None     | None     | None     | None     |
| 8   | 30             | 4            | 1/7 1/7 1/7 1/7 0/2            | 1           | 1           | 1           | 1           | 0           | 30        | 7.5     | 1/7 1.5/7 2/7 1/2              | 1      | 1.5    | 2      | 1      | None   | 30          | 5.5       | 2/7 1.5/7 0/7 2/7 0/2          | 2        | 1.5      | 0        | 2        | 0        |
| 9   | 23             | 4            | None                           | None        | None        | None        | None        | None        | 23        | 7       | None                           | None   | None   | None   | None   | None   | 23          | 5         | None                           | None     | None     | None     | None     | None     |
| 10  | 22             | 9            | None                           | None        | None        | None        | None        | None        | 22        | 11      | None                           | None   | None   | None   | None   | None   | 22          | 9         | None                           | None     | None     | None     | None     | None     |