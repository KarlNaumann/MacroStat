===============================
Behavioral Equations GL06INSOUT
===============================
--------------
Step Equations
--------------
1. Actual Inventory Sales Ratio

Calculate the actual inventory-sales ratio (beginning-of-period).

.. math::
	:label: actual_inventory_sales_ratio
	:nowrap:

	\begin{align}
	\sigma(t) = inv(t-1) / s(t)
	\end{align}


2. Advances Demand

Calculate bank demand for central bank advances.
When the tentative liquidity ratio is below floor, bank borrows from CB.

.. math::
	:label: advances_demand
	:nowrap:

	\begin{align}
	A^d(t) = \max(bot \cdot (M1_s + M2_s) - B_b^T, 0) \cdot z_3
	\end{align}


3. Bank Bill Holdings

Calculate actual bank bill holdings after advances.

.. math::
	:label: bank_bill_holdings
	:nowrap:

	\begin{align}
	B_b(t) = B_b^T(t) + A^d(t)
	\end{align}


4. Bank Bills Tentative

Calculate tentative bank bill holdings (residual of bank balance sheet).

.. math::
	:label: bank_bills_tentative
	:nowrap:

	\begin{align}
	B_b^T(t) = M1_s + M2_s - L_s - Hb_s
	\end{align}


5. Bank Liquidity Ratio

Calculate the actual bank liquidity ratio.

.. math::
	:label: bank_liquidity_ratio
	:nowrap:

	\begin{align}
	BLR(t) = B_b(t) / (M1_s(t) + M2_s(t))
	\end{align}


6. Bank Liquidity Ratio Tentative

Calculate the tentative bank liquidity ratio.

.. math::
	:label: bank_liquidity_ratio_tentative
	:nowrap:

	\begin{align}
	BLR^T(t) = B_b^T(t) / (M1_s(t) + M2_s(t))
	\end{align}


7. Bank Profit Margin

Calculate the bank profit margin as a share of the prior deposit base.

.. math::
	:label: bank_profit_margin
	:nowrap:

	\begin{align}
	BPM(t) = \frac{FBP(t) + FBP(t-1)}{M1_s(t-1) + M1_s(t-2)
	+ M2_s(t-1) + M2_s(t-2)}
	Uses LaggedM1Supply(t-1)=M1_s(t-1) and prior LaggedM1Supply(t-2)=M1_s(t-2).
	\end{align}


8. Bank Profits

Calculate bank profits from prior-period stocks and rates.

.. math::
	:label: bank_profits
	:nowrap:

	\begin{align}
	FBP(t) = r_l(t-1) \cdot L_s(t-1) + r_b(t-1) \cdot B_b(t-1)
	- r_m(t-1) \cdot M2_s(t-1) - r_a(t-1) \cdot A_s(t-1)
	\end{align}


9. Bills Central Bank

Calculate central bank bill holdings as residual.

.. math::
	:label: bills_central_bank
	:nowrap:

	\begin{align}
	B_{cb}(t) = B_s(t) - B_h(t) - B_b(t)
	\end{align}


10. Bills Demand

Calculate bills demand via Tobin portfolio choice.

.. math::
	:label: bills_demand
	:nowrap:

	\begin{align}
	B^d(t) = V^e_{nc} \cdot (\lambda_{30} + \lambda_{31} \cdot r_m
	+ \lambda_{32} \cdot r_b + \lambda_{33} \cdot ERr_{bl})
	+ \lambda_{34} \cdot YD^e
	\end{align}


11. Bills Supply

Calculate bill supply as residual government financing.

.. math::
	:label: bills_supply
	:nowrap:

	\begin{align}
	B_s(t) = B_s(t-1) + PSBR(t)
	- (BL_s(t) - BL_s(t-1)) \cdot p_{bl}(t)
	\end{align}


12. Bond Price

Calculate the bond price as the inverse of the bond yield.

.. math::
	:label: bond_price
	:nowrap:

	\begin{align}
	p_{bl}(t) = \frac{1}{r_{bl}(t)}
	\end{align}


13. Bonds Demand

Calculate bond demand (in number of bonds) via Tobin portfolio choice.

.. math::
	:label: bonds_demand
	:nowrap:

	\begin{align}
	BL^d(t) = \frac{V^e_{nc} \cdot (\lambda_{40} + \lambda_{41} \cdot r_m
	+ \lambda_{42} \cdot r_b + \lambda_{43} \cdot ERr_{bl})
	+ \lambda_{44} \cdot YD^e}{p_{bl}(t)}
	\end{align}


14. Bonds Supply

Set bond supply equal to household bond demand.

.. math::
	:label: bonds_supply
	:nowrap:

	\begin{align}
	BL_s(t) = BL_h(t)
	\end{align}


15. Capital Gains

Calculate capital gains on bond holdings.

.. math::
	:label: capital_gains
	:nowrap:

	\begin{align}
	CG(t) = (p_{bl}(t) - p_{bl}(t-1)) \cdot BL_h(t-1)
	\end{align}


16. Cash Demand

Calculate household cash demand proportional to consumption.

.. math::
	:label: cash_demand
	:nowrap:

	\begin{align}
	Hh^d(t) = \lambda_c \cdot C(t)
	\end{align}


17. Central Bank Profits

Calculate central bank profits from prior-period stocks and rates.

.. math::
	:label: central_bank_profits
	:nowrap:

	\begin{align}
	FCB(t) = r_b(t-1) \cdot B_{cb}(t-1) + r_a(t-1) \cdot A_s(t-1)
	\end{align}


18. Deposit Rate

Calculate the endogenous deposit rate based on bank liquidity.

.. math::
	:label: deposit_rate
	:nowrap:

	\begin{align}
	r_m(t) = r_m(t-1) + \zeta_m \cdot (z_4 - z_5)
	+ \zeta_b \cdot (r_b(t) - r_b(t-1))
	where :math:`z_4 = 1` if :math:`BLR^T(t-1) < bot`,
	\end{align}


19. Employment

Calculate employment from real output and labor productivity.

.. math::
	:label: employment
	:nowrap:

	\begin{align}
	N(t) = \frac{y(t)}{pr}
	\end{align}


20. Expected Inventories

Calculate expected end-of-period inventories via partial adjustment.

.. math::
	:label: expected_inventories
	:nowrap:

	\begin{align}
	inv^e(t) = inv(t-1) - \gamma \cdot (inv(t-1) - inv^T(t))
	\end{align}


21. Expected Nominal Wealth

Calculate expected nominal wealth for portfolio allocation.

.. math::
	:label: expected_nominal_wealth
	:nowrap:

	\begin{align}
	V^e(t) = V(t-1) + YD^e(t) - C(t)
	\end{align}


22. Expected Non Cash Wealth

Calculate expected non-cash wealth available for portfolio allocation.

.. math::
	:label: expected_non_cash_wealth
	:nowrap:

	\begin{align}
	V^e_{nc}(t) = V^e(t) - Hh^d(t)
	\end{align}


23. Expected Real Disposable Income

Calculate expected real disposable income via adaptive expectations.

.. math::
	:label: expected_real_disposable_income
	:nowrap:

	\begin{align}
	yd_r^e(t) = \varepsilon \cdot yd_r(t-1)
	+ (1 - \varepsilon) \cdot yd_r^e(t-1)
	\end{align}


24. Expected Return On Bonds

Calculate the expected return on bonds (static expectations).

.. math::
	:label: expected_return_on_bonds
	:nowrap:

	\begin{align}
	ERr_{bl}(t) = r_{bl}(t)
	\end{align}


25. Expected Sales

Calculate expected real sales using adaptive expectations.

.. math::
	:label: expected_sales
	:nowrap:

	\begin{align}
	s^e(t) = \beta \cdot s(t-1) + (1 - \beta) \cdot s^e(t-1)
	\end{align}


26. Firm Profits

Calculate firm profits.

.. math::
	:label: firm_profits
	:nowrap:

	\begin{align}
	FP(t) = S(t) - TX(t) - WB(t) + (INV(t) - INV(t-1))
	- r_l(t-1) \cdot L_s(t-1)
	\end{align}


27. Government Debt

Calculate total government debt.

.. math::
	:label: government_debt
	:nowrap:

	\begin{align}
	GD(t) = B_s(t) + p_{bl}(t) \cdot BL_s(t)
	\end{align}


28. Government Spending

Calculate nominal government spending.

.. math::
	:label: government_spending
	:nowrap:

	\begin{align}
	G(t) = g(t) \cdot p(t)
	\end{align}


29. Haig Simons Disposable Income

Calculate Haig-Simons disposable income (including capital gains).

.. math::
	:label: haig_simons_disposable_income
	:nowrap:

	\begin{align}
	YD_{hs}(t) = YD_r(t) + CG(t)
	\end{align}


30. High Powered Money

Calculate high-powered money supply.

.. math::
	:label: high_powered_money
	:nowrap:

	\begin{align}
	H_s(t) = Hh(t) + Hb_s(t)
	\end{align}


31. Inflation Rate

Calculate the inflation rate.

.. math::
	:label: inflation_rate
	:nowrap:

	\begin{align}
	\pi(t) = \frac{p(t) - p(t-1)}{p(t-1)}
	\end{align}


32. Loan Demand

Set firm loan demand equal to nominal inventories.

.. math::
	:label: loan_demand
	:nowrap:

	\begin{align}
	L^d(t) = INV(t)
	\end{align}


33. Loan Rate

Calculate the endogenous loan rate based on bank profit margin.

.. math::
	:label: loan_rate
	:nowrap:

	\begin{align}
	r_l(t) = r_l(t-1) + \zeta_l \cdot (z_6 - z_7)
	+ (r_b(t) - r_b(t-1))
	where :math:`z_6 = 1` if :math:`BPM(t) < bot_{pm}`,
	\end{align}


34. M1 Demand Tentative

Calculate tentative M1 demand as the portfolio residual.
M1 is the buffer-stock asset that absorbs whatever *actual* wealth
remains after the three Tobin-allocated assets (M2, Bills, Bonds).
The R sfcr reference uses actual non-cash wealth (``Vnc = V - Hhh``),
not expected, for this residual.

.. math::
	:label: m1_demand_tentative
	:nowrap:

	\begin{align}
	M1^{dt}(t) = V_{nc}(t) - M2^d(t) - B^d(t) - p_{bl}(t) \cdot BL^d(t)
	\end{align}


35. M2 Demand

Calculate M2 demand via Tobin portfolio choice.

.. math::
	:label: m2_demand
	:nowrap:

	\begin{align}
	M2^d(t) = V^e_{nc} \cdot (\lambda_{20} + \lambda_{21} \cdot r_m
	+ \lambda_{22} \cdot r_b + \lambda_{23} \cdot ERr_{bl})
	+ \lambda_{24} \cdot YD^e
	\end{align}


36. M2 Household

Calculate household M2 holdings.
When z1=1 (normal), M2 = M2Demand. When z2=1, M2 absorbs the
residual non-cash wealth after bills and bonds.

.. math::
	:label: m2_household
	:nowrap:

	\begin{align}
	M2_h(t) = M2^d \cdot z_1
	+ (V_{nc} - B_h - p_{bl} \cdot BL^d) \cdot z_2
	\end{align}


37. Nominal Consumption

Calculate nominal consumption.

.. math::
	:label: nominal_consumption
	:nowrap:

	\begin{align}
	C(t) = c(t) \cdot p(t)
	\end{align}


38. Nominal Expected Disposable Income

Calculate the nominal expected disposable income.

.. math::
	:label: nominal_expected_disposable_income
	:nowrap:

	\begin{align}
	YD^e(t) = yd_r^e(t) \cdot p(t) + \pi(t) \cdot \frac{V(t-1)}{p(t)}
	\end{align}


39. Nominal Inventories

Calculate nominal inventories valued at current unit cost.

.. math::
	:label: nominal_inventories
	:nowrap:

	\begin{align}
	INV(t) = inv(t) \cdot UC(t)
	\end{align}


40. Nominal Output

Calculate nominal output (sales at price + inventory change at cost).

.. math::
	:label: nominal_output
	:nowrap:

	\begin{align}
	Y(t) = p(t) \cdot s(t) + UC(t) \cdot (inv(t) - inv(t-1))
	\end{align}


41. Nominal Sales

Calculate nominal sales.

.. math::
	:label: nominal_sales
	:nowrap:

	\begin{align}
	S(t) = s(t) \cdot p(t)
	\end{align}


42. Nominal Wage

Calculate the nominal wage via partial adjustment to target real wage.

.. math::
	:label: nominal_wage
	:nowrap:

	\begin{align}
	W(t) = W(t-1) \cdot (1 + \Omega_3 \cdot (\omega^T(t-1) - \omega(t-1)))
	where :math:`\omega(t-1) = W(t-1) / p(t-1)`.
	\end{align}


43. Nominal Wealth

Calculate nominal household wealth.

.. math::
	:label: nominal_wealth
	:nowrap:

	\begin{align}
	V(t) = V(t-1) + YD_{hs}(t) - C(t)
	\end{align}


44. Non Cash Wealth

Calculate actual non-cash wealth available for portfolio allocation.
In the R sfcr reference, M1 (the buffer-stock residual) is computed
against actual non-cash wealth (``Vnc = V - Hhh``), not expected.
Portfolio *demands* (M2d, Bhd, BLd) use expected wealth (VncE), but the
realized residual uses actual wealth so that the portfolio identity holds.

.. math::
	:label: non_cash_wealth
	:nowrap:

	\begin{align}
	V_{nc}(t) = V(t) - Hh^d(t)
	\end{align}


45. Normal Historic Unit Cost

Calculate the normal historic unit cost.
A weighted average of current and prior unit cost, with inventory
financing cost applied to the prior component.

.. math::
	:label: normal_historic_unit_cost
	:nowrap:

	\begin{align}
	NHUC(t) = (1 - \sigma^T) \cdot UC(t)
	+ \sigma^T \cdot (1 + r_l(t-1)) \cdot UC(t-1)
	\end{align}


46. Price Level

Calculate the price level as a markup over normal historic unit cost.

.. math::
	:label: price_level
	:nowrap:

	\begin{align}
	p(t) = (1 + \tau) \cdot (1 + \phi) \cdot NHUC(t)
	\end{align}


47. Public Sector Borrowing Requirement

Calculate the public sector borrowing requirement.

.. math::
	:label: public_sector_borrowing_requirement
	:nowrap:

	\begin{align}
	PSBR(t) = G(t) + r_b(t-1) \cdot B_s(t-1) + BL_s(t-1)
	- T(t) - FCB(t)
	\end{align}


48. Real Consumption

Calculate real consumption.

.. math::
	:label: real_consumption
	:nowrap:

	\begin{align}
	c(t) = \alpha_0 + \alpha_1 \cdot yd_r^e(t) + \alpha_2 \cdot v(t-1)
	\end{align}


49. Real Inventories

Calculate real end-of-period inventories.

.. math::
	:label: real_inventories
	:nowrap:

	\begin{align}
	inv(t) = inv(t-1) + y(t) - s(t)
	\end{align}


50. Real Output

Calculate real output as expected sales plus inventory adjustment.

.. math::
	:label: real_output
	:nowrap:

	\begin{align}
	y(t) = s^e(t) + inv^e(t) - inv(t-1)
	\end{align}


51. Real Regular Disposable Income

Calculate real regular disposable income with inflation erosion.

.. math::
	:label: real_regular_disposable_income
	:nowrap:

	\begin{align}
	yd_r(t) = \frac{YD_r(t)}{p(t)} - \pi(t) \cdot \frac{V(t-1)}{p(t)}
	\end{align}


52. Real Sales

Calculate real sales as consumption plus real government spending.

.. math::
	:label: real_sales
	:nowrap:

	\begin{align}
	s(t) = c(t) + g(t)
	\end{align}


53. Real Wage

Calculate the current real wage.

.. math::
	:label: real_wage
	:nowrap:

	\begin{align}
	\omega(t) = W(t) / p(t)
	\end{align}


54. Real Wealth

Calculate real household wealth.

.. math::
	:label: real_wealth
	:nowrap:

	\begin{align}
	v(t) = V(t) / p(t)
	\end{align}


55. Regular Disposable Income

Calculate regular (non-capital-gains) disposable income.
With indirect taxation, households receive firm/bank dividends (which
already net out the production tax) plus interest income. No explicit
tax subtraction at the household level.

.. math::
	:label: regular_disposable_income
	:nowrap:

	\begin{align}
	YD_r(t) = WB(t) + FD(t)
	+ r_m(t-1) \cdot M2_h(t-1) + r_b(t-1) \cdot B_h(t-1)
	+ BL_h(t-1)
	\end{align}


56. Required Reserves

Calculate required reserves.

.. math::
	:label: required_reserves
	:nowrap:

	\begin{align}
	RR(t) = ro_1 \cdot M1_s(t) + ro_2 \cdot M2_s(t)
	\end{align}


57. Target Inventories

Calculate target inventories.

.. math::
	:label: target_inventories
	:nowrap:

	\begin{align}
	inv^T(t) = \sigma^T(t) \cdot s^e(t)
	\end{align}


58. Target Inventory Sales Ratio

Calculate the target inventory-sales ratio.

.. math::
	:label: target_inventory_sales_ratio
	:nowrap:

	\begin{align}
	\sigma^T(t) = \sigma_0 - \sigma_1 \cdot r_l(t)
	\end{align}


59. Target Real Wage

Calculate the target real wage from productivity and employment.

.. math::
	:label: target_real_wage
	:nowrap:

	\begin{align}
	\omega^T(t) = \exp(\Omega_0 + \Omega_1 \log(pr)
	+ \Omega_2 \log(N(t) / N_{fe}))
	\end{align}


60. Taxes

Calculate indirect production tax on sales.

.. math::
	:label: taxes
	:nowrap:

	\begin{align}
	TX(t) = \frac{\tau}{1 + \tau} \cdot S(t)
	\end{align}


61. Total Dividends

Calculate total dividends distributed to households.
All firm and bank profits are distributed as dividends.

.. math::
	:label: total_dividends
	:nowrap:

	\begin{align}
	FD(t) = FP(t) + FBP(t)
	\end{align}


62. Unit Cost

Calculate the unit cost of production.

.. math::
	:label: unit_cost
	:nowrap:

	\begin{align}
	UC(t) = \frac{WB(t)}{y(t)}
	\end{align}


63. Wage Bill

Calculate the nominal wage bill.

.. math::
	:label: wage_bill
	:nowrap:

	\begin{align}
	WB(t) = N(t) \cdot W(t)
	\end{align}
