from simulation.wez import WEZEstimator


def main():

    estimator = WEZEstimator(

        shooter_altitude_ft=25000,

        shooter_speed_knots=500,

        shooter_pitch_deg=10,

        target_altitude_ft=22000,

        target_speed_knots=450,

        target_heading_deg=30,

        target_off_boresight_deg=20,

    )

    rmax = estimator.find_rmax()

    print()

    print("=" * 50)

    print("WEAPON ENGAGEMENT ZONE")

    print("=" * 50)

    print()

    print(f"Maximum Launch Range : {rmax:.2f} meters")

    print(f"Maximum Launch Range : {rmax/1000:.2f} km")

    print()

    print("=" * 50)


if __name__ == "__main__":
    main()