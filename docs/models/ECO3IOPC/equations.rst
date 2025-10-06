=============================
Behavioral Equations ECO3IOPC
=============================
--------------
Step Equations
--------------
1. Carbon Mass Nonrenewable Energy

The carbon mass of non-renewable energy is given by the conversion
of emissions (due to non-renewable energy) with a fixed constant

.. math::
	:label: carbon_mass_nonrenewable_energy
	:nowrap:

	\begin{align}
	cen(t) &= \frac{emis(t)}{car}
	\end{align}




2. Central Bank Bill Holdings

Calculate the central bank bill holdings.

.. math::
	:label: central_bank_bill_holdings
	:nowrap:

	\begin{align}
	B_{CB}(t) = B_{s}(t) - B_{h}(t)
	\end{align}




3. Central Bank Money Stock

Calculate the central bank money stock.

.. math::
	:label: central_bank_money_stock
	:nowrap:

	\begin{align}
	H_{s}(t) = H_{s}(t-1) + (B_{CB}(t) - B_{CB}(t-1))
	\end{align}




4. Central Bank Profits

Calculate the central bank profits (income on bills held).

.. math::
	:label: central_bank_profits
	:nowrap:

	\begin{align}
	r(t-1)B_{CB}(t-1)
	\end{align}




5. Co2 Intensity Change

The energy emission intensity decreases by a fixed percentage
each period

.. math::
	:label: co2_intensity_change
	:nowrap:

	\begin{align}
	\beta_e(t) = \beta_e(t-1) (1 - \Delta_\% \beta_e)
	\end{align}




6. Consumption

Calculate the consumption.

.. math::
	:label: consumption
	:nowrap:

	\begin{align}
	c(t) = \alpha_1 \left(\frac{YD^e(t)}{p_c(t)} - \pi(t)\right) + \alpha_2 \frac{V(t-1)}{p_c(t)}
	\end{align}




7. Cumulative Co2 Emissions

Cumulative CO2 emissions are simply incremented by the current
emissions

.. math::
	:label: cumulative_co2_emissions
	:nowrap:

	\begin{align}
	co2_{cum}(t) = co2_{cum}(t-1) + emis(t)
	\end{align}




8. Discarding Of Socioeconomic Stock

The discarding of socioeconomic stock occurs as a percentage of
existing stock, converted into units of matter

.. math::
	:label: discarding_of_socioeconomic_stock
	:nowrap:

	\begin{align}
	dis(t) &= m_{mat}^\top (\zeta \cdot dc(t-1))
	\end{align}




9. Disposable Income

Calculate the disposable income.

.. math::
	:label: disposable_income
	:nowrap:

	\begin{align}
	YD(t) = Y(t) - T(t) + r(t-1)B_h(t-1)
	\end{align}




10. Emissions From Nonrenewable Energy

Emissions are based on the use of non-renewable energy, with a fixed
emission intensity

.. math::
	:label: emissions_from_nonrenewable_energy
	:nowrap:

	\begin{align}
	emis(t) = \beta_e nen(t)
	\end{align}




11. Energy Reserves

Energyreserves are depleted by human use and incremented by the
conversion from resources

.. math::
	:label: energy_reserves
	:nowrap:

	\begin{align}
	k_e(t) &= k_e(t-1) + conv_e(t) - mat(t)
	\end{align}




12. Energy To Resource Conversion

Energy resources are converted into reserves at a fixed rate

.. math::
	:label: energy_to_resource_conversion
	:nowrap:

	\begin{align}
	res_e(t) &= res_e(t-1) - conv_e(t)\\
	conv_e(t) &= \sigma_e res_e(t)
	\end{align}




13. Energy Used In Production

Energy use in production is given by a fixed energy intensity of
production

.. math::
	:label: energy_used_in_production
	:nowrap:

	\begin{align}
	en(t) = \epsilon_e^\top x(t)
	\end{align}




14. Expected Disposable Income

The expected disposable income is simply the prior period's
disposable income. Equation (3.20) in the book.

.. math::
	:label: expected_disposable_income
	:nowrap:

	\begin{align}
	YD^e(t) = YD(t-1)
	\end{align}




15. Expected Wealth

Calculate the expected wealth.

.. math::
	:label: expected_wealth
	:nowrap:

	\begin{align}
	V^e(t) = V(t-1) + YD^e(t) - C(t)
	\end{align}




16. Extraction Of Matter

The matter extracted is the difference in the matter consumed and
the matter that was recycled

.. math::
	:label: extraction_of_matter
	:nowrap:

	\begin{align}
	mat(t) &= x_{mat} - rec(t)
	\end{align}




17. Final Demand

Calculate the final demand as the sum of household and government
demands spread over the sectors

.. math::
	:label: final_demand
	:nowrap:

	\begin{align}
	d_i(t) = \beta_{HH,i}C_{HH}(t) + \beta_{GOV,i}G(t)
	\end{align}




18. Government Bill Issuance

Calculate the government bill issuance.

.. math::
	:label: government_bill_issuance
	:nowrap:

	\begin{align}
	B_s(t) = B_s(t-1) + (G(t) - r(t-1)B_s(t-1)) - (T(t) + r(t-1)B_{CB}(t-1))
	\end{align}




19. Household Bill Demand

Calculate the household bill demand.

.. math::
	:label: household_bill_demand
	:nowrap:

	\begin{align}
	\frac{B_h(t)}{V^e(t)} = \lambda_0 + \lambda_1 r(t) - \lambda_2 \frac{YD^e(t)}{V^e(t)}
	\end{align}




20. Household Bill Holdings

Calculate the household bill holdings.

.. math::
	:label: household_bill_holdings
	:nowrap:

	\begin{align}
	B_h(t) = B_h(t-1) + (B_h^d(t) - B_h(t-1))
	\end{align}




21. Household Money Stock

Calculate the household deposits as a residual.

.. math::
	:label: household_money_stock
	:nowrap:

	\begin{align}
	H_h(t) = V(t) - B_h(t)
	\end{align}




22. Inflation

Compute the inflation (i.e. term for absence of money illusion)

.. math::
	:label: inflation
	:nowrap:

	\begin{align}
	\pi(t) &= \left(\frac{p_c(t) - p_c(t-1)}{p_c(t-1)}\right)\left(\frac{V(t-1)}{p_c(t-1)}\right)
	\end{align}




23. Interest Earned On Bills Household

Calculate the interest earned on bills by the household.

.. math::
	:label: interest_earned_on_bills_household
	:nowrap:

	\begin{align}
	r(t-1)B_h(t-1)
	\end{align}




24. Material Goods Production

The material goods production in the economy

.. math::
	:label: material_goods_production
	:nowrap:

	\begin{align}
	x_{mat}(t) &= m_{mat}^\top x(t)
	\end{align}




25. Matter Reserves

Matter reserves are depleted by human use and incremented by the
conversion from resources

.. math::
	:label: matter_reserves
	:nowrap:

	\begin{align}
	k_m(t) &= k_m(t-1) + conv_m(t) - mat(t)
	\end{align}




26. Matter To Resource Conversion

Matter resources is converted into reserves at a fixed rate

.. math::
	:label: matter_to_resource_conversion
	:nowrap:

	\begin{align}
	res(t) &= res(t-1) - conv_m(t)\\
	conv_m(t) &= \sigma_m res(t)
	\end{align}




27. National Income

National income is the sum of nominal final demand

.. math::
	:label: national_income
	:nowrap:

	\begin{align}
	Y(t) = P^\top(t)d(t)
	\end{align}




28. Non Renewable Energy Used In Production

Non-renewable energy use in production is given by the difference in
energy used and renewable energy used.

.. math::
	:label: non_renewable_energy_used_in_production
	:nowrap:

	\begin{align}
	nen(t) = en(t) - ren(t)
	\end{align}




29. Oxygen

The oxygen level is given by the difference in emissions and the
carbon mass of energy

.. math::
	:label: oxygen
	:nowrap:

	\begin{align}
	o2(t) &= emis(t) - cen(t)
	\end{align}




30. Price Indices

Compute the consumer and government price indices based on their
consumption shares

.. math::
	:label: price_indices
	:nowrap:

	\begin{align}
	p_c(t) &= \beta_{HH}^\top P(t)\\
	p_g(t) &= \beta_{G}^\top P(t)
	\end{align}




31. Prices

Compute the sectoral prices as the sum of unit labour cost and a
markup on intermediate prices

.. math::
	:label: prices
	:nowrap:

	\begin{align}
	P_i(t) = \frac{w}{pr_i} + (1 + \mu)\sum_j a_{ij}P_j(t)
	\end{align}




32. Propensity To Consume Income

Endogenous propensity to consume out of income, dependent on the
rate of interest

.. math::
	:label: propensity_to_consume_income
	:nowrap:

	\begin{align}
	\alpha_1(t) = \alpha_{10} - \alpha_{11} r(t-1)
	\end{align}




33. Real Gross Output

Compute real gross output as the solution to the linear set of
equations

.. math::
	:label: real_gross_output
	:nowrap:

	\begin{align}
	x(t) = (I - A)^{-1}d(t)
	\end{align}




34. Recycling Of Discarded Stock

A fixed share of the discarded socioeconomic stock is recycled

.. math::
	:label: recycling_of_discarded_stock
	:nowrap:

	\begin{align}
	rec(t) &= \rho_{dis} dis(t)
	\end{align}




35. Renewable Energy Used In Production

Renewable energy use in production is given by a fixed energy intensity of
production combined with a fixed share of energy sourced from renewables

.. math::
	:label: renewable_energy_used_in_production
	:nowrap:

	\begin{align}
	ren(t) = \epsilon_e^\top (\eta_{en} \cdot x(t))
	\end{align}




36. Set Interest Rate

Set the interest rate. This is given exogenously by the scenario.

.. math::
	:label: set_interest_rate
	:nowrap:

	\begin{align}
	r(t) = \bar{r}
	\end{align)
	\end{align}




37. Socioeconomic Stock

The socioeconomic stock grows through material extraction and
shrinks due to discards

.. math::
	:label: socioeconomic_stock
	:nowrap:

	\begin{align}
	k_h(t) &= k_h(t-1) + x_{mat}(t) - dis(t)
	\end{align}




38. Stock Of Durable Goods

The stock of durable goods evolves based on inflows from consumption
and outflows from discard

.. math::
	:label: stock_of_durable_goods
	:nowrap:

	\begin{align}
	dc(t) &= dc(t-1) + B_c c(t) - \zeta dc(t-1)
	\end{align}




39. Taxes

Calculate the taxes.

.. math::
	:label: taxes
	:nowrap:

	\begin{align}
	T(t) = \theta (Y(t) + r(t-1)B_h(t-1))
	\end{align}




40. Temperature

Temperature is determined by a transformation of cumulative CO2

.. math::
	:label: temperature
	:nowrap:

	\begin{align}
	temp(t) = \frac{1}{1-fnc}\cdot tcre \cdot co2_{cum}(t)
	\end{align}




41. Waste

Waste is computed as the difference in matter extraction and the
growth in the SocioeconomicStock

.. math::
	:label: waste
	:nowrap:

	\begin{align}
	wa(t) &= mat(t) - (k_h(t) - k_h(t-1))
	\end{align}




42. Wealth

Calculate the wealth.

.. math::
	:label: wealth
	:nowrap:

	\begin{align}
	V(t) = V(t-1) + YD(t) - C(t)
	\end{align}
