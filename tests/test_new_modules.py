"""
Quick test script to verify the new modules work correctly.

Run this from the project root:
    python test_new_modules.py
"""

print("=" * 60)
print("Testing New QuantFinance Modules")
print("=" * 60)

# Test 1: StochasticProcesses imports
print("\n1. Testing StochasticProcesses imports...")
try:
    from StochasticProcesses import GBM, CIRProcess, OrnsteinUhlenbeck, HestonModel, BDT
    print("   ✓ All stochastic process classes imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")

# Test 2: FixedIncome imports
print("\n2. Testing FixedIncome imports...")
try:
    from FixedIncome import Bond, CallableBond, YieldCurve
    print("   ✓ All fixed income classes imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")

# Test 3: Create and price a simple bond
print("\n3. Testing Bond class...")
try:
    from FixedIncome import Bond
    bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
    price = bond.price(market_rate=0.04)
    ytm = bond.yield_to_maturity(price)
    duration = bond.duration(market_rate=0.04)
    print(f"   ✓ Bond pricing works")
    print(f"     - Price at 4% yield: ${price:.2f}")
    print(f"     - YTM: {ytm:.4%}")
    print(f"     - Duration: {duration:.4f} years")
except Exception as e:
    print(f"   ✗ Bond class failed: {e}")

# Test 4: Simulate GBM
print("\n4. Testing GBM simulation...")
try:
    from StochasticProcesses import GBM
    import numpy as np

    gbm = GBM(initial_value=100, mu=0.05, sigma=0.2, random_seed=42)
    t, paths = gbm.simulate(T=1.0, n_steps=252, n_paths=100)

    expected = gbm.expected_value(1.0)
    empirical = np.mean(paths[:, -1])

    print(f"   ✓ GBM simulation works")
    print(f"     - Expected final value: ${expected:.2f}")
    print(f"     - Empirical mean: ${empirical:.2f}")
    print(f"     - Paths shape: {paths.shape}")
except Exception as e:
    print(f"   ✗ GBM simulation failed: {e}")

# Test 5: YieldCurve with interpolation
print("\n5. Testing YieldCurve with interpolation...")
try:
    from FixedIncome import YieldCurve
    from FixedIncome.interpolation import CubicSplineInterpolator

    yc = YieldCurve(
        tenors=[0.25, 0.5, 1, 2, 5, 10, 30],
        yields=[0.025, 0.028, 0.032, 0.035, 0.038, 0.040, 0.042]
    )

    interpolator = CubicSplineInterpolator()
    yc.set_interpolator(interpolator)

    yield_7y = yc.get_yield(7.0)
    df_5y = yc.get_discount_factor(5.0)
    metrics = yc.get_shape_metrics()

    print(f"   ✓ YieldCurve works")
    print(f"     - 7-year yield: {yield_7y:.4%}")
    print(f"     - 5-year discount factor: {df_5y:.6f}")
    print(f"     - 2-10 slope: {metrics['slope_2_10']:.4%}")
except Exception as e:
    print(f"   ✗ YieldCurve failed: {e}")

# Test 6: CIR process with Feller condition
print("\n6. Testing CIR process...")
try:
    from StochasticProcesses import CIRProcess
    import numpy as np

    cir = CIRProcess(initial_value=0.05, theta=0.5, mu=0.04, sigma=0.02, random_seed=42)

    feller = cir.satisfies_feller_condition()
    expected = cir.expected_value(5.0)
    variance = cir.variance(5.0)

    t, paths = cir.simulate(T=5.0, n_steps=100, n_paths=100, method='exact')
    empirical_mean = np.mean(paths[:, -1])

    print(f"   ✓ CIR process works")
    print(f"     - Feller condition satisfied: {feller}")
    print(f"     - Expected value at T=5: {expected:.4%}")
    print(f"     - Empirical mean: {empirical_mean:.4%}")
except Exception as e:
    print(f"   ✗ CIR process failed: {e}")

# Test 7: BDT model
print("\n7. Testing Black-Derman-Toy model...")
try:
    from StochasticProcesses import BDT

    bdt = BDT(r0=0.05, volatilities=[0.02, 0.021, 0.022, 0.023], dt=1.0)
    tree = bdt.build_tree()
    zcb_price = bdt.price_zero_coupon_bond(face_value=100)

    print(f"   ✓ BDT model works")
    print(f"     - Tree shape: {tree.shape}")
    print(f"     - Zero-coupon bond price: ${zcb_price:.2f}")
except Exception as e:
    print(f"   ✗ BDT model failed: {e}")

# Test 8: Integration - Bond pricing with CIR
print("\n8. Testing integration (Bond + CIR pricing)...")
try:
    from FixedIncome import Bond
    from FixedIncome.pricing import BondPricerWithIRModel
    from StochasticProcesses import CIRProcess

    bond = Bond(face_value=1000, coupon_rate=0.05, maturity=5, frequency=1)
    cir = CIRProcess(initial_value=0.05, theta=0.5, mu=0.04, sigma=0.01, random_seed=42)

    pricer = BondPricerWithIRModel(cir)
    result = pricer.price(bond, n_paths=1000, n_steps=50)

    # Compare with flat rate
    flat_price = bond.price(market_rate=0.04)

    print(f"   ✓ Integration works")
    print(f"     - CIR Monte Carlo price: ${result['price']:.2f} ± ${result['std_error']:.2f}")
    print(f"     - Flat rate price: ${flat_price:.2f}")
    print(f"     - Difference: ${abs(result['price'] - flat_price):.2f}")
except Exception as e:
    print(f"   ✗ Integration failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)
print("\nIf all tests passed, the reorganization was successful.")
print("You can now use these modules in your notebooks and scripts.")
print("\nNext steps:")
print("  1. Try importing in a Jupyter notebook")
print("  2. Run your existing notebooks to ensure compatibility")
print("  3. See REORGANIZATION_SUMMARY.md for detailed information")
