=========================
Behavioral Equations IOPC
=========================
--------------
Step Equations
--------------
1. Central Bank Bill Holdings

Calculate the central bank bill holdings.

.. math::
	:label: central_bank_bill_holdings
	:nowrap:

	\begin{align}
	B_{CB}(t) = B_{s}(t) - B_{h}(t)
	\end{align}




2. Central Bank Money Stock

Calculate the central bank money stock.

.. math::
	:label: central_bank_money_stock
	:nowrap:

	\begin{align}
	H_{s}(t) = H_{s}(t-1) + (B_{CB}(t) - B_{CB}(t-1))
	\end{align}




3. Central Bank Profits

Calculate the central bank profits (income on bills held).

.. math::
	:label: central_bank_profits
	:nowrap:

	\begin{align}
	r(t-1)B_{CB}(t-1)
	\end{align}




4. Consumption

Calculate the consumption.

.. math::
	:label: consumption
	:nowrap:

	\begin{align}
	c(t) = \alpha_1 \left(\frac{YD^e(t)}{p_c(t)} - \pi(t)\right) + \alpha_2 \frac{V(t-1)}{p_c(t)}
	\end{align}




5. Disposable Income

Calculate the disposable income.

.. math::
	:label: disposable_income
	:nowrap:

	\begin{align}
	YD(t) = Y(t) - T(t) + r(t-1)B_h(t-1)
	\end{align}




6. Expected Disposable Income

The expected disposable income is simply the prior period's
disposable income. Equation (3.20) in the book.

.. math::
	:label: expected_disposable_income
	:nowrap:

	\begin{align}
	YD^e(t) = YD(t-1)
	\end{align}




7. Expected Wealth

Calculate the expected wealth.

.. math::
	:label: expected_wealth
	:nowrap:

	\begin{align}
	V^e(t) = V(t-1) + YD^e(t) - C(t)
	\end{align}




8. Final Demand

Calculate the final demand as the sum of household and government
demands spread over the sectors

.. math::
	:label: final_demand
	:nowrap:

	\begin{align}
	d_i(t) = \beta_{HH,i}C_{HH}(t) + \beta_{GOV,i}G(t)
	\end{align}




9. Government Bill Issuance

Calculate the government bill issuance.

.. math::
	:label: government_bill_issuance
	:nowrap:

	\begin{align}
	B_s(t) = B_s(t-1) + (G(t) - r(t-1)B_s(t-1)) - (T(t) + r(t-1)B_{CB}(t-1))
	\end{align}




10. Household Bill Demand

Calculate the household bill demand.

.. math::
	:label: household_bill_demand
	:nowrap:

	\begin{align}
	\frac{B_h(t)}{V^e(t)} = \lambda_0 + \lambda_1 r(t) - \lambda_2 \frac{YD^e(t)}{V^e(t)}
	\end{align}




11. Household Bill Holdings

Calculate the household bill holdings.

.. math::
	:label: household_bill_holdings
	:nowrap:

	\begin{align}
	B_h(t) = B_h(t-1) + (B_h^d(t) - B_h(t-1))
	\end{align}




12. Household Money Stock

Calculate the household deposits as a residual.

.. math::
	:label: household_money_stock
	:nowrap:

	\begin{align}
	H_h(t) = V(t) - B_h(t)
	\end{align}




13. Inflation

Compute the inflation (i.e. term for absence of money illusion)

.. math::
	:label: inflation
	:nowrap:

	\begin{align}
	\pi(t) &= \left(\frac{p_c(t) - p_c(t-1)}{p_c(t-1)}\right)\left(\frac{V(t-1)}{p_c(t-1)}\right)
	\end{align}




14. Interest Earned On Bills Household

Calculate the interest earned on bills by the household.

.. math::
	:label: interest_earned_on_bills_household
	:nowrap:

	\begin{align}
	r(t-1)B_h(t-1)
	\end{align}




15. National Income

National income is the sum of nominal final demand

.. math::
	:label: national_income
	:nowrap:

	\begin{align}
	Y(t) = P^\top(t)d(t)
	\end{align}




16. Price Indices

Compute the consumer and government price indices based on their
consumption shares

.. math::
	:label: price_indices
	:nowrap:

	\begin{align}
	p_c(t) &= \beta_{HH}^\top P(t)\\
	p_g(t) &= \beta_{G}^\top P(t)
	\end{align}




17. Prices

Compute the sectoral prices as the sum of unit labour cost and a
markup on intermediate prices

.. math::
	:label: prices
	:nowrap:

	\begin{align}
	P_i(t) = \frac{w}{pr_i} + (1 + \mu)\sum_j a_{ij}P_j(t)
	\end{align}




18. Propensity To Consume Income

Endogenous propensity to consume out of income, dependent on the
rate of interest

.. math::
	:label: propensity_to_consume_income
	:nowrap:

	\begin{align}
	\alpha_1(t) = \alpha_{10} - \alpha_{11} r(t-1)
	\end{align}




19. Real Gross Output

Compute real gross output as the solution to the linear set of
equations

.. math::
	:label: real_gross_output
	:nowrap:

	\begin{align}
	x(t) = (I - A)^{-1}d(t)
	\end{align}




20. Set Interest Rate

Set the interest rate. This is given exogenously by the scenario.

.. math::
	:label: set_interest_rate
	:nowrap:

	\begin{align}
	r(t) = \bar{r}
	\end{align)
	\end{align}




21. Taxes

Calculate the taxes.

.. math::
	:label: taxes
	:nowrap:

	\begin{align}
	T(t) = \theta (Y(t) + r(t-1)B_h(t-1))
	\end{align}




22. Wealth

Calculate the wealth.

.. math::
	:label: wealth
	:nowrap:

	\begin{align}
	V(t) = V(t-1) + YD(t) - C(t)
	\end{align}
