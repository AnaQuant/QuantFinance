"""
Unit tests for FixedIncome/bonds.py.

Covers the standalone functional interface (bond_price, compute_dv01,
compute_convexity) and the Bond OO class (price, duration,
modified_duration, convexity).  Each test validates a specific financial
property so failures surface economically meaningful regressions rather
than opaque numerical discrepancies.
"""

import pytest

from FixedIncome.bonds import Bond, bond_price, compute_dv01, compute_convexity
from tests.conftest import FACE, COUPON, YTM, MATURITY


# ===========================================================================
# bond_price()
# ===========================================================================


class TestBondPrice:
    """Tests for the standalone bond_price() functional helper."""

    def test_at_par_when_coupon_equals_ytm(self):
        """
        When coupon rate equals YTM the bond must price at par.

        Financial rationale: every coupon exactly compensates the investor
        for the opportunity cost of holding the bond, so the market pays
        neither a premium nor a discount.  Tolerance of ±0.01 accommodates
        floating-point rounding across ten discount periods.
        """
        rate = 0.05
        price = bond_price(FACE, rate, rate, MATURITY)
        assert price == pytest.approx(FACE, abs=0.01), (
            f"At-par pricing failed: coupon=ytm={rate}, expected ~{FACE}, got {price:.6f}"
        )

    def test_below_par_when_ytm_exceeds_coupon(self):
        """
        When YTM > coupon rate the bond must price below par (discount bond).

        Financial rationale: the fixed coupons are insufficient to meet the
        market's required return, so the price must fall to make up the
        difference through capital appreciation at maturity.
        """
        high_ytm = COUPON + 0.01   # 5 % YTM vs 4 % coupon
        price = bond_price(FACE, COUPON, high_ytm, MATURITY)
        assert price < FACE, (
            f"Discount bond pricing failed: ytm={high_ytm} > coupon={COUPON}, "
            f"expected price < {FACE}, got {price:.6f}"
        )

    def test_above_par_when_coupon_exceeds_ytm(self):
        """
        When coupon rate > YTM the bond must price above par (premium bond).

        Financial rationale: the coupons are more generous than the market
        rate, so investors bid the price above face value.  The premium
        erodes to par at maturity (pull-to-par effect).
        """
        low_ytm = COUPON - 0.01    # 3 % YTM vs 4 % coupon
        price = bond_price(FACE, COUPON, low_ytm, MATURITY)
        assert price > FACE, (
            f"Premium bond pricing failed: coupon={COUPON} > ytm={low_ytm}, "
            f"expected price > {FACE}, got {price:.6f}"
        )


# ===========================================================================
# compute_dv01()
# ===========================================================================


class TestComputeDV01:
    """Tests for the compute_dv01() functional helper."""

    def test_dv01_is_positive(self):
        """
        DV01 must always be positive for a plain-vanilla fixed-coupon bond.

        Financial rationale: bond prices and yields move inversely —
        a 1bp rise in yield always reduces the bond's present value.
        DV01 = P(y) − P(y+1bp), so it must be strictly positive.
        """
        dv01 = compute_dv01(FACE, COUPON, YTM, MATURITY)
        assert dv01 > 0, (
            f"DV01 must be positive; got {dv01:.8f} "
            f"(face={FACE}, coupon={COUPON}, ytm={YTM}, years={MATURITY})"
        )

    def test_dv01_increases_with_maturity(self):
        """
        A 10-year bond must have higher DV01 than a 3-year bond with the
        same coupon and yield.

        Financial rationale: longer-dated bonds have more cash flows
        discounted at distant horizons, amplifying their sensitivity to
        yield changes.  This is the fundamental risk/duration relationship.
        """
        dv01_3yr = compute_dv01(FACE, COUPON, YTM, 3)
        dv01_10yr = compute_dv01(FACE, COUPON, YTM, MATURITY)
        assert dv01_10yr > dv01_3yr, (
            f"DV01 should increase with maturity: "
            f"dv01(10yr)={dv01_10yr:.8f} vs dv01(3yr)={dv01_3yr:.8f}"
        )

    def test_dv01_scales_linearly_with_face_value(self):
        """
        Doubling the face value must (approximately) double DV01.

        Financial rationale: DV01 is a linear function of notional — a
        £200 position has exactly twice the rate sensitivity of a £100
        position in the same instrument.  Relative tolerance of 0.1 %
        covers floating-point rounding.
        """
        dv01_100 = compute_dv01(100.0, COUPON, YTM, MATURITY)
        dv01_200 = compute_dv01(200.0, COUPON, YTM, MATURITY)
        assert dv01_200 == pytest.approx(2 * dv01_100, rel=0.001), (
            f"DV01 linear scaling failed: dv01(200)={dv01_200:.8f}, "
            f"2×dv01(100)={2 * dv01_100:.8f}"
        )


# ===========================================================================
# compute_convexity()
# ===========================================================================


class TestComputeConvexity:
    """Tests for the compute_convexity() functional helper."""

    def test_convexity_is_positive(self):
        """
        Convexity must be positive for a plain-vanilla fixed-coupon bond.

        Financial rationale: the price-yield curve is convex (bowed
        outward) for standard bonds — prices rise more when yields fall
        than they fall when yields rise by the same amount.  This
        second-order effect is always non-negative for bullet bonds.
        """
        convexity = compute_convexity(FACE, COUPON, YTM, MATURITY)
        assert convexity > 0, (
            f"Convexity must be positive; got {convexity:.6f} "
            f"(face={FACE}, coupon={COUPON}, ytm={YTM}, years={MATURITY})"
        )

    def test_convexity_increases_with_maturity(self):
        """
        A 10-year bond must have higher convexity than a 3-year bond with
        the same coupon and yield.

        Financial rationale: longer maturities produce greater curvature
        in the price-yield relationship because cash flows at distant
        horizons are discounted with higher-order effects, making the
        bond's response to large yield moves increasingly non-linear.
        """
        convexity_3yr = compute_convexity(FACE, COUPON, YTM, 3)
        convexity_10yr = compute_convexity(FACE, COUPON, YTM, MATURITY)
        assert convexity_10yr > convexity_3yr, (
            f"Convexity should increase with maturity: "
            f"convexity(10yr)={convexity_10yr:.4f} vs convexity(3yr)={convexity_3yr:.4f}"
        )


# ===========================================================================
# Bond class
# ===========================================================================


class TestBondClass:
    """Tests for the Bond OO class methods."""

    def test_price_matches_standalone_bond_price(self, bond: Bond):
        """
        Bond.price() must agree with standalone bond_price() for equivalent
        inputs (annual frequency, same face/coupon/ytm/maturity).

        Financial rationale: both implementations use identical present-value
        mathematics.  Divergence would indicate a formula error in one path.
        Tolerance of 1e-8 reflects acceptable floating-point accumulation
        over ten discount periods.
        """
        class_price = bond.price(YTM)
        func_price = bond_price(FACE, COUPON, YTM, MATURITY)
        assert class_price == pytest.approx(func_price, abs=1e-8), (
            f"Bond.price() and bond_price() disagree: "
            f"class={class_price:.8f}, functional={func_price:.8f}"
        )

    def test_duration_positive_and_less_than_maturity(self, bond: Bond):
        """
        Macaulay duration must be strictly positive and strictly less than
        time to maturity for a coupon-bearing bond.

        Financial rationale: duration is the present-value-weighted average
        life of the bond's cash flows.  Because intermediate coupons are
        received before final redemption, the weighted average must be
        shorter than the full maturity.  A zero would indicate no cash-flow
        weighting; a value ≥ maturity would imply all weight is on the
        final cash flow (zero-coupon behaviour).
        """
        mac_dur = bond.duration(YTM)
        assert mac_dur > 0, (
            f"Macaulay duration must be positive; got {mac_dur:.6f}"
        )
        assert mac_dur < bond.maturity, (
            f"Macaulay duration must be < maturity for a coupon bond: "
            f"duration={mac_dur:.6f}, maturity={bond.maturity}"
        )

    def test_modified_duration_less_than_macaulay_duration(self, bond: Bond):
        """
        Modified duration must be strictly less than Macaulay duration.

        Financial rationale: modified duration = Macaulay duration /
        (1 + y/freq).  Because the divisor is always > 1 for positive
        yields, modified duration is always a fraction of Macaulay
        duration.  This relationship is the bridge between the time-
        weighted measure and the price-sensitivity measure.
        """
        mac_dur = bond.duration(YTM)
        mod_dur = bond.modified_duration(YTM)
        assert mod_dur < mac_dur, (
            f"Modified duration must be < Macaulay duration: "
            f"modified={mod_dur:.6f}, macaulay={mac_dur:.6f}"
        )

    def test_convexity_positive(self, bond: Bond):
        """
        Bond.convexity() must be positive for a plain-vanilla coupon bond.

        Financial rationale: identical to the standalone test — the
        analytical formula used by the Bond class (sum of t*(t+1)-weighted
        discounted cash flows) is positive by construction for all
        non-negative cash flows and positive yields.  A negative result
        would reveal a sign or indexing error in the formula.
        """
        convexity = bond.convexity(YTM)
        assert convexity > 0, (
            f"Bond.convexity() must be positive; got {convexity:.6f} "
            f"(face={bond.face_value}, coupon={bond.coupon_rate}, ytm={YTM})"
        )
