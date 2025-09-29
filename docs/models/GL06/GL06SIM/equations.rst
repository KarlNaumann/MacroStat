============================
Behavioral Equations GL06SIM
============================
------------------------
Initialization Equations
------------------------
Initialize the behavior of the Godley-Lavoie 2006 SIM model.
Within the book the initialization is generally to set all non-scenario
variables to zero.

.. math::
	:label: initialize
	:nowrap:

	\begin{align}
	C_d(0) &= C_s(0) = 0 \\
	G_d(0) &= G_s(0) = 0 \\
	T_s(0) &= T_d(0) = 0 \\
	N_s(0) &= N_d(0) = 0 \\
	YD(0) &= 0 \\
	W(0) &= 0 \\
	H_s(0) &= 0 \\
	H_h(0) &= 0
	\end{align}


--------------
Step Equations
--------------
1. Consumption Demand

The consumption demand is a function of the disposable income,
the propensity to consume income, and the propensity to consume savings.
Equation (3.7) in the book.

.. math::
	:label: consumption_demand
	:nowrap:

	\begin{align}
	C_d(t) = \alpha_1 YD(t) + \alpha_2 H_h(t-1)
	\end{align}




2. Consumption Supply

In the model it is assumed that the supply will adjust to the demand,
that is, whatever is demanded can and will be produced. Equation (3.1)
in the book.

.. math::
	:label: consumption_supply
	:nowrap:

	\begin{align}
	C_s(t) = C_d(t)
	\end{align}




3. Disposable Income

The disposable income is the wage bill minus the taxes.
Equation (3.5) in the book.

.. math::
	:label: disposable_income
	:nowrap:

	\begin{align}
	YD(t) = W(t) N_s(t) - T_s(t)
	\end{align}




4. Government Money Stock

The government money stock is a function of the government demand,
and the tax supply. Equation (3.8) in the book.

.. math::
	:label: government_money_stock
	:nowrap:

	\begin{align}
	H_s(t) = H_s(t-1) + G_d(t) - T_d(t)
	\end{align}




5. Government Supply

In the model it is assumed that the supply will adjust to the demand,
that is, whatever is demanded can and will be produced. Equation (3.2)
in the book.

.. math::
	:label: government_supply
	:nowrap:

	\begin{align}
	G_s(t) = G_d(t)
	\end{align}




6. Household Money Stock

The household money stock is a function of the disposable income,
the propensity to consume income, and the propensity to consume savings.
Equation (3.9) in the book.

.. math::
	:label: household_money_stock
	:nowrap:

	\begin{align}
	H_h(t) = H_h(t-1) + YD(t) - C_d(t)
	\end{align}




7. Labour Demand

We can resolve the labour demand from the national income equation,
together with the consumption demand (+ disposable income) and the government demand
knowing that labour demand is equal to labour supply.

.. math::
	:label: labour_demand
	:nowrap:

	\begin{align}
	N_d(t) =\frac{\alpha_2 H_h(t-1) + G_d}{W(t)(1-\alpha_1(1-\theta))}
	\end{align}




8. Labour Income

The labour income is the wage rate times the labour supply. This is
an intermediate variable used to calculate the disposable income, but is
computed explicitly here to compute the transaction flows.

.. math::
	:label: labour_income
	:nowrap:

	\begin{align}
	W(t) N_s(t)
	\end{align}




9. Labour Supply

In the model it is assumed that the supply will be equal to
the amount of labour demanded. Equation (3.4) in the book

.. math::
	:label: labour_supply
	:nowrap:

	\begin{align}
	N_s(t) = N_d(t)
	\end{align}




10. National Income

The national income is the sum of the consumption demand,
the government demand, and the tax supply. Equation (3.10) in the book.

.. math::
	:label: national_income
	:nowrap:

	\begin{align}
	Y(t) = C_s(t) + G_s(t)
	\end{align}




11. Tax Demand

The tax demand is a function of the tax rate, the labour supply,
and the wage rate. Equation (3.6) in the book.

.. math::
	:label: tax_demand
	:nowrap:

	\begin{align}
	T_d(t) = \theta N_s(t) W(t)
	\end{align}




12. Tax Supply

In the model it is assumed that the supply will be equal to
the amount of taxes demanded. Equation (3.3) in the book

.. math::
	:label: tax_supply
	:nowrap:

	\begin{align}
	T_s(t) = T_d(t)
	\end{align}
