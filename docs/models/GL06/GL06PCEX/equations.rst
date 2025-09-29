=============================
Behavioral Equations GL06PCEX
=============================
------------------------
Initialization Equations
------------------------
Initialize the behavior of the Godley-Lavoie 2006 PCEX model.
Within the book the initialization is generally to set all non-scenario
variables to zero. Accordingly

.. math::
	:label: initialize
	:nowrap:

	\begin{align}
	C(0) &= 0 \\
	G(0) &= 0 \\
	Y(0) &= 0 \\
	T(0) &= 0 \\
	YD(0) &= 0 \\
	V(0) &= 0 \\
	H_s(0) &= 0 \\
	H_h(0) &= 0 \\
	B_h(0) &= 0 \\
	B_s(0) &= 0 \\
	B_{CB}(0) &= 0 \\
	r(0) &= 0 \\
	\end{align}


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
	C(t) = \alpha_1 YD^e(t) + \alpha_2 V(t-1)
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




8. Government Bill Issuance

Calculate the government bill issuance.

.. math::
	:label: government_bill_issuance
	:nowrap:

	\begin{align}
	B_s(t) = B_s(t-1) + (G(t) - r(t-1)B_s(t-1)) - (T(t) + r(t-1)B_{CB}(t-1))
	\end{align}




9. Household Bill Demand

Calculate the household bill demand.

.. math::
	:label: household_bill_demand
	:nowrap:

	\begin{align}
	\frac{B_h(t)}{V^e(t)} = \lambda_0 + \lambda_1 r(t) - \lambda_2 \frac{YD^e(t)}{V^e(t)}
	\end{align}




10. Household Bill Holdings

Calculate the household bill holdings.

.. math::
	:label: household_bill_holdings
	:nowrap:

	\begin{align}
	B_h(t) = B_h(t-1) + (B_h^d(t) - B_h(t-1))
	\end{align}




11. Household Money Stock

Calculate the household deposits as a residual.

.. math::
	:label: household_money_stock
	:nowrap:

	\begin{align}
	H_h(t) = V(t) - B_h(t)
	\end{align}




12. Interest Earned On Bills Household

Calculate the interest earned on bills by the household.

.. math::
	:label: interest_earned_on_bills_household
	:nowrap:

	\begin{align}
	r(t-1)B_h(t-1)
	\end{align}




13. National Income

Calculate the national income based on the closed-form solution derived in the documentation.
The closed-form solution is used to avoid the need to solve the system of equations iteratively, thus
preserving the differentiability of the model trajectory.

.. math::
	:label: national_income
	:nowrap:

	\begin{align}
	Y(t) = C(t) + G(t)
	\end{align}




14. Taxes

Calculate the taxes.

.. math::
	:label: taxes
	:nowrap:

	\begin{align}
	T(t) = \theta (Y(t) + r(t-1)B_h(t-1))
	\end{align}




15. Wealth

Calculate the wealth.

.. math::
	:label: wealth
	:nowrap:

	\begin{align}
	V(t) = V(t-1) + YD(t) - C(t)
	\end{align}
