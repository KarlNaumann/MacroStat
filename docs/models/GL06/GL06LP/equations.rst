===========================
Behavioral Equations GL06LP
===========================
------------------------
Initialization Equations
------------------------
Initialize the behavior of the Godley-Lavoie 2006 LP model.
All non-scenario variables are set to zero, consistent with the
standard SFC initialization approach.

.. math::
	:label: initialize
	:nowrap:

	\begin{align}
	C(0) = G(0) = Y(0) = T(0) = YD_r(0) = 0 \\
	V(0) = H_h(0) = H_s(0) = B_h(0) = B_s(0) = 0 \\
	B_{CB}(0) = BL_h(0) = BL_s(0) = 0 \\
	r_b(0) = p_{bl}(0) = r_{bl}(0) = 0
	\end{align}


--------------
Step Equations
--------------
1. Bond Coupon Income Household

Calculate the coupon income from bonds held by the household.
Each bond pays a coupon of 1 monetary unit per period, so bond
coupon income is simply the number of bonds held last period.

.. math::
	:label: bond_coupon_income_household
	:nowrap:

	\begin{align}
	BL_h(t-1)
	\end{align}




2. Bond Yield

Calculate the bond yield (coupon / price).
Since the coupon is 1 per bond, the yield is simply the inverse
of the bond price.

.. math::
	:label: bond_yield
	:nowrap:

	\begin{align}
	r_{bl}(t) = \frac{1}{p_{bl}(t)}
	\end{align}




3. Capital Gains

Calculate the capital gains on bonds.
Capital gains arise from changes in the bond price applied to the
stock of bonds held at the beginning of the period.

.. math::
	:label: capital_gains
	:nowrap:

	\begin{align}
	CG(t) = \left(p_{bl}(t) - p_{bl}(t-1)\right) \cdot BL_h(t-1)
	\end{align}




4. Central Bank Bill Holdings

Calculate the central bank bill holdings.

.. math::
	:label: central_bank_bill_holdings
	:nowrap:

	\begin{align}
	B_{CB}(t) = B_s(t) - B_h(t)
	\end{align}




5. Central Bank Money Stock

Calculate the central bank money stock.

.. math::
	:label: central_bank_money_stock
	:nowrap:

	\begin{align}
	H_s(t) = H_s(t-1) + (B_{CB}(t) - B_{CB}(t-1))
	\end{align}




6. Central Bank Profits

Calculate the central bank profits (income on bills held).

.. math::
	:label: central_bank_profits
	:nowrap:

	\begin{align}
	r_b(t-1) \cdot B_{CB}(t-1)
	\end{align}




7. Consumption

Calculate consumption.

.. math::
	:label: consumption
	:nowrap:

	\begin{align}
	C(t) = \alpha_1 \cdot YD_r^e(t) + \alpha_2 \cdot V(t-1)
	\end{align}




8. Disposable Income

Calculate the disposable income.
Disposable income includes national income minus taxes plus
interest on bills and bond coupon income.

.. math::
	:label: disposable_income
	:nowrap:

	\begin{align}
	YD_r(t) = Y(t) - T(t) + r_b(t-1) \cdot B_h(t-1) + BL_h(t-1)
	\end{align}




9. Expected Bond Price

Calculate the expected bond price.
In the base LP model, agents have static expectations: the
expected bond price equals the current bond price. This method
is designed to be overridden in LP2/LP3 for adaptive expectations.

.. math::
	:label: expected_bond_price
	:nowrap:

	\begin{align}
	p_{bl}^e(t) = p_{bl}(t)
	\end{align}




10. Expected Capital Gains

Calculate the expected capital gains on bonds.
In the base LP model with static expectations (pebl = pbl),
expected capital gains are zero.

.. math::
	:label: expected_capital_gains
	:nowrap:

	\begin{align}
	CG^e(t) = \chi \cdot (p_{bl}^e(t) - p_{bl}(t))
	\cdot BL_h(t)
	\end{align}




11. Expected Disposable Income

Calculate the expected disposable income.
Expectations are adaptive: expected disposable income equals the
prior period's actual disposable income.

.. math::
	:label: expected_disposable_income
	:nowrap:

	\begin{align}
	YD_r^e(t) = YD_r(t-1)
	\end{align}




12. Expected Return On Bonds

Calculate the expected return on bonds.
The expected return combines the current yield with the expected
capital gain from bond price changes.

.. math::
	:label: expected_return_on_bonds
	:nowrap:

	\begin{align}
	ERr_{bl}(t) = r_{bl}(t)
	+ \chi \cdot \frac{p_{bl}^e(t) - p_{bl}(t)}{p_{bl}(t)}
	\end{align}




13. Expected Wealth

Calculate the expected wealth.
Expected wealth uses expected disposable income instead of
actual disposable income, but actual capital gains (which are
known at the start of the period since bond prices are
exogenous).

.. math::
	:label: expected_wealth
	:nowrap:

	\begin{align}
	V^e(t) = V(t-1) + (YD_r^e(t) - C(t)) + CG(t)
	\end{align}




14. Government Bill Issuance

Calculate the government bill issuance.
The government budget constraint determines the supply of bills:
new bills finance the deficit net of bond issuance revenue.

.. math::
	:label: government_bill_issuance
	:nowrap:

	\begin{align}
	B_s(t) = B_s(t-1)
	+ G(t) + r_b(t-1) \cdot B_s(t-1) + BL_s(t-1)
	- T(t) - r_b(t-1) \cdot B_{CB}(t-1)
	- (BL_s(t) - BL_s(t-1)) \cdot p_{bl}(t)
	\end{align}




15. Government Bond Supply

Calculate the government bond supply.
Bond supply equals bond demand from households.

.. math::
	:label: government_bond_supply
	:nowrap:

	\begin{align}
	BL_s(t) = BL_h(t)
	\end{align}




16. Household Bill Demand

Calculate the household bill demand using Tobin portfolio choice.

.. math::
	:label: household_bill_demand
	:nowrap:

	\begin{align}
	B_d(t) = V^e(t) \cdot \lambda_{20}
	+ V^e(t) \cdot (\lambda_{22} \cdot r_b(t)
	+ \lambda_{23} \cdot ERr_{bl}(t))
	+ \lambda_{24} \cdot YD_r^e(t)
	\end{align}




17. Household Bill Holdings

Calculate the household bill holdings.

.. math::
	:label: household_bill_holdings
	:nowrap:

	\begin{align}
	B_h(t) = B_d(t)
	\end{align}




18. Household Bond Demand

Calculate the household bond demand using Tobin portfolio choice.
The demand is in number of bonds (value / price).

.. math::
	:label: household_bond_demand
	:nowrap:

	\begin{align}
	BL_d(t) = \frac{V^e(t) \cdot \left(\lambda_{30}
	+ \lambda_{32} \cdot r_b(t)
	+ \lambda_{33} \cdot ERr_{bl}(t)
	+ \lambda_{34} \cdot \frac{YD_r^e(t)}{V^e(t)}\right)}
	{p_{bl}(t)}
	\end{align}




19. Household Bond Holdings

Calculate the household bond holdings.

.. math::
	:label: household_bond_holdings
	:nowrap:

	\begin{align}
	BL_h(t) = BL_d(t)
	\end{align}




20. Household Cash Demand

Calculate the household cash demand as a residual.

.. math::
	:label: household_cash_demand
	:nowrap:

	\begin{align}
	H_d(t) = V^e(t) - B_d(t) - p_{bl}(t) \cdot BL_d(t)
	\end{align}




21. Household Cash Stock

Calculate the household cash stock as a residual.
Cash is what remains of wealth after bills and bonds.

.. math::
	:label: household_cash_stock
	:nowrap:

	\begin{align}
	H_h(t) = V(t) - B_h(t) - p_{bl}(t) \cdot BL_h(t)
	\end{align}




22. Interest On Bills Household

Calculate the interest earned on bills by the household.

.. math::
	:label: interest_on_bills_household
	:nowrap:

	\begin{align}
	r_b(t-1) \cdot B_h(t-1)
	\end{align}




23. National Income

Calculate the national income.

.. math::
	:label: national_income
	:nowrap:

	\begin{align}
	Y(t) = C(t) + G(t)
	\end{align}




24. Taxes

Calculate the taxes.
The tax base includes national income, interest on bills, and
bond coupon income.

.. math::
	:label: taxes
	:nowrap:

	\begin{align}
	T(t) = \theta \cdot \left(Y(t) + r_b(t-1) \cdot B_h(t-1)
	+ BL_h(t-1)\right)
	\end{align}




25. Wealth

Calculate the wealth.
Wealth evolves as savings plus capital gains on bonds.

.. math::
	:label: wealth
	:nowrap:

	\begin{align}
	V(t) = V(t-1) + (YD_r(t) - C(t)) + CG(t)
	\end{align}
