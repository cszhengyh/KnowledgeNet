import re
import sys
import openai
sys.path.append("/share/project/zhengyuhui/domain_tree_3.1.1")
from chat_with_gpt import chat_with_gpt

messages = []
openai.api_key = "EMPTY"
openai.base_url = "http://127.0.0.1:7999/v1/"
model = "Llama3-70B-merge-v1-0613-ep3"

label_prompt = r"""
You are a senior assistant in the field of knowledge extraction. You will be provided with a text and multiple knowledge. Your task is to determine what knowledge the text involves and explain step by step why the text involves this knowledge respectively. The text can involve multiple knowledge and the knowledge involved in the text must all be in the knowledge provided. You shouldn't determine what knowledge the text involves by guessing or reasoning, but instead base it on what the text actually says. Deliver your response in this format: [labels: ......, Explanation: ......]. If the text does not involve any of the provided knowledge, your response should be [labels: none, Explanation: (here you should indicate what knowledge involved by the text and state that it is not in the knowledge provided)]. For example, the multiple knowledge categories are "B, C, D, E", the text is "A" and A involves knowledge B, C, your response should be [labels: B, C, Explanation: (here you should explain step by step why A involves knowledge B and knowledge C respectively)].
"""

sft_data_instruction = """
You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.q: A money lender lends Rs.2000 for six months at 20% p.a. rate. It the interest is reckoned quarterly than find the amount given after time limit.
Options:
(A) 2205
(B) 3387
(C) 2882
(D) 2997
(E) 1278
a: Stream of consciousness: A = 2000(21/20)2 = 2205 The answer is (A).
--
q: In 2005, 30 percent of the math department’s 40 students were female, and in 2007, 40 percent of the math department’s 65 students were female. What was the percent increase from 2005 to 2007 in the number of female students in the department?
Options:
(A) 15%
(B) 116%
(C) 136%
(D) 155%
(E) 160%
a: Stream of consciousness: Female students in 2005 = 30% of 40 = 12
Female students in 2007 = 40% of 65 = 26
Percentage increase = [(26 - 12)/12] *100 % = 116% The answer is (B).
--
q: If x<0, y<0, and |x| > |y|, which of the following must be true?
Options:
(A) x > y
(B) y^2 > x^2
(C) x^3 > y^2
(D) x < y
(E) x < –y
a:Stream of consciousness: Both x and y are negative, but the absolute value of x is greater than the absolute value of y. This means that x is "more negative" than y.

(A) x > y: This is not true, as x is more negative than y.
(B) y^2 > x^2: Since both x and y are negative, their squares will be positive. Since |x| > |y|, x^2 > y^2, so this option is not true.
(C) x^3 > y^2: When we cube a negative number, it remains negative. So, x^3 will be negative. Since x and y are both negative, their squares will be positive. Therefore, y^2 will be positive. A negative number (x^3) cannot be greater than a positive number (y^2), so this option is not true.
(D) x < y: This is true because x is more negative than y.
(E) x < -y: Since x is negative and y is negative, -y will be positive. A negative number (x) will always be less than a positive number (-y), so this is true.

Options (D) and (E) must be true.
"""

labels = """
Elementary Number Theory, Analytic Number Theory, Algebraic Number Theory, Transcendental Number Theory, Diophantine Approximation, Arithmetic Geometry, Probabilistic Number Theory, Computational Number Theory, Geometric Probability, Probability Distribution, Extreme Value Theory, Stochastic Process, Markov Process, Stochastic Analysis, Martingale Theory, Applied Probability Theory, Linear Programming, Nonlinear Programming, Dynamic Programming, Combinatorial Optimization, Parametric Programming, Integer Programming, Stochastic Programming, Queueing Theory, Game Theory, Inventory Theory, Decision Theory, Search Theory, Graph Theory, Operational Research, Optimization, Foundations of Mathematics, Linear Algebra, Group Theory, Field Theory, Lie group, Lie Algebra, Kac-Moody Algebra, Ring Theory, Module Theory, Lattice Theory, Universal Algebra, Category Theory, Homological Algebra, Algebraic K-theory, Differential Algebra, Algebraic Coding Theory, Sampling Theory, Hypothesis Testing, Nonparametric Statistics, Analysis of Variance, Correlation and Regression Analysis, Statistical Inference, Bayesian Statistics, Design of Experiments, Multivariate Analysis, Statistical Decision Theory, Time Series Analysis, Spatial Statistics, Deductive Logic (Symbolic Logic), Proof Theory (Metamathematics), Recursion Theory, Model Theory, Axiomatic Set Theory, Statistical Quality Control, Reliability Mathematics, Actuarial Science, Computational Statistics, Statistical Simulation, Differential Calculus, Calculus, Theory of Series, Numerical Solution of Ordinary Differential Equations, Numerical Solution of Partial Differential Equations, Numerical Methods for Integral Transforms and Integral Equations, Numerical Algebra, Computational Optimization Methods, Approximation Theory and Computational Geometry, Stochastic Numerical Methods and Statistical Computing, Parallel Computing Algorithms, Error Analysis and Interval Algorithms, Numerical Methods for Wavelet and Fourier Analysis, Computational Methods for Inverse Problems, Symbolic Computation and Automated Reasoning
"""

if __name__ == "__main__":
    messages = [{"role":"user","content": label_prompt + "\n\n" + f"the text is \"{sft_data_instruction}\", the multiple knowledge categories are \"{labels}\". Please respond following the above requirements and examples."}]
    response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.95
            )        
    targets = response.choices[0].message.content    
    print(targets)