#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <algorithm>
#include <iterator>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <typename number>
  void
  LaserData<number>::add_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("laser");
    {
      prm.add_parameter(
        "model",
        model,
        "Laser model. "
        "analytical_temperature: see Mirkoohi et al. (2019); "
        "volumetric: volumetric heat source, the intensity is defined by \"\"intensity profile\"\"; "
        "interface_projection: projection-based regularized continuum surface flux in \"direction\", the intensity is defined by \"\"intensity profile\"\"; "
        "interface_projection_sharp: projection-based sharp surface flux in \"direction\", the intensity is defined by \"\"intensity profile\"\"; "
        "interface_projection_sharp_conforming: projection-based sharp surface flux in \"direction\" on a conforming mesh, the intensity is defined by \"\"intensity profile\"\"; "
        "RTE: continuum surface flux projected using the radiative transport equation in \"direction\", supporting shadowing of undercuts, the intensity is defined by \"\"intensity profile\"\"; ");
      prm.add_parameter(
        "intensity profile",
        intensity_profile,
        "Laser intensity profile. "
        "uniform: note that the \"power\" input is treated as the uniform power density in the whole domain; "
        "Gauss: Gaussian laser intensity shape with \"radius\" that retains the \"power\"; "
        "Gusarov: see Gusarov et al. (2009); ");
      prm.add_parameter("power", power, "Laser power");
      prm.add_parameter("power over time",
                        power_over_time,
                        "Temporal distribution of the laser power",
                        Patterns::Selection("constant|ramp"));
      prm.add_parameter("power start time",
                        power_start_time,
                        "In case of time-dependent laser power: activation time of ");
      prm.add_parameter("power end time",
                        power_end_time,
                        "In case of time-dependent laser power: end time of ");

      prm.add_parameter("absorptivity gas",
                        absorptivity_gas,
                        "Laser energy absorptivity of the gaseous part of the domain.");
      prm.add_parameter("absorptivity liquid",
                        absorptivity_liquid,
                        "Laser energy absorptivity of the liquid part of the domain.");
      prm.add_parameter(
        "do move",
        do_move,
        "Set this parameter to true to move the laser in x-direction with the given parameter scan speed.");
      prm.add_parameter(
        "starting position",
        starting_position,
        "Center coordinates of the laser beam starting position on the interface melt/gas.");
      prm.add_parameter("scan speed", scan_speed, "Scan speed of the laser");
      prm.add_parameter("direction", direction, "Laser beam direction.");
      prm.add_parameter("radius", radius, "Laser beam radius.");
      /*
       *   Gusarov
       */
      prm.enter_subsection("gusarov");
      {
        prm.add_parameter("reflectivity", gusarov.reflectivity, "Reflectivity of the material.");
        prm.add_parameter("extinction coefficient",
                          gusarov.extinction_coefficient,
                          "Extinction coefficient in [1/m].");
        prm.add_parameter("layer thickness", gusarov.layer_thickness, "Layer thickness");
      }
      prm.leave_subsection();
      /*
       *   Analytical temperature field
       */
      prm.enter_subsection("analytical");
      {
        prm.add_parameter("ambient temperature",
                          analytical.ambient_temperature,
                          "Ambient temperature in the inert gas.");
        prm.add_parameter(
          "max temperature",
          analytical.max_temperature,
          "Maximum temperature arising in the melt pool. If this temperature is lower than the boiling"
          " temperature, this value is corrected to correspond to the boiling temperature + 500 K.");
        prm.add_parameter("temperature x to y ratio",
                          analytical.temperature_x_to_y_ratio,
                          "This factor scales the analytical temperature field to be anisotropic.");
      }
      prm.leave_subsection();

      delta_approximation_phase_weighted.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  LaserData<number>::post(
    const unsigned int          dim,
    const bool                  heat_use_volume_specific_thermal_capacity_for_phase_interpolation,
    const MaterialData<number> &material)
  {
    // if the laser starting position is not specified, set it to the origin
    if (starting_position.size() == 0)
      {
        starting_position.resize(dim);
        std::fill(starting_position.begin(), starting_position.end(), 0);
      }
    else
      AssertThrow(starting_position.size() == dim,
                  ExcMessage(
                    "There must be dim coordinates of the laser starting position given."));

    // if the laser direction is not specified, set it to the negative dim-1 direction
    if (direction.size() == 0)
      {
        direction.resize(dim);
        std::fill(direction.begin(), std::prev(direction.end()), 0);
        direction[dim - 1] = -1.0;
      }
    else
      {
        AssertThrow(direction.size() == dim,
                    ExcMessage("There must be dim coordinates of the laser direction given."));
        AssertThrow(std::any_of(direction.begin(),
                                direction.end(),
                                [](double d) { return d != 0.0; }),
                    ExcMessage("The laser direction cannot be a zero vector."));
      }

    // set automatic weights of asymmetric delta functions, if requested
    if (heat_use_volume_specific_thermal_capacity_for_phase_interpolation)
      delta_approximation_phase_weighted.set_parameters(
        material, LevelSet::ParameterScaledInterpolationType::volume_specific_heat_capacity);
    else
      delta_approximation_phase_weighted.set_parameters(
        material, LevelSet::ParameterScaledInterpolationType::specific_heat_capacity_times_density);
  }

  template <typename number>
  void
  LaserData<number>::check_input_parameters() const
  {
    if (power <= 0.0)
      return;
    if (model == LaserModelType::analytical_temperature)
      return;

    if (intensity_profile != LaserIntensityProfileType::uniform)
      AssertThrow(radius > 0.0, ExcMessage("The radius cannot be zero!"));
    if (intensity_profile == LaserIntensityProfileType::Gusarov)
      AssertThrow(model == LaserModelType::volumetric,
                  ExcMessage("For the Gusarov laser profile, the laser model must be volumetric!"));
  }

  template <typename number>
  template <int dim>
  Point<dim, number>
  LaserData<number>::get_starting_position() const
  {
    return UtilityFunctions::to_point<dim>(starting_position.begin(), starting_position.end());
  }

  template <typename number>
  template <int dim>
  Tensor<1, dim, number>
  LaserData<number>::get_direction() const
  {
    const auto temp = UtilityFunctions::to_point<dim>(direction.begin(), direction.end());
    return temp / temp.norm();
  }

  template struct LaserData<double>;

  template Point<1, double>
  LaserData<double>::get_starting_position<1>() const;
  template Point<2, double>
  LaserData<double>::get_starting_position<2>() const;
  template Point<3, double>
  LaserData<double>::get_starting_position<3>() const;
  template Tensor<1, 1, double>
  LaserData<double>::get_direction<1>() const;
  template Tensor<1, 2, double>
  LaserData<double>::get_direction<2>() const;
  template Tensor<1, 3, double>
  LaserData<double>::get_direction<3>() const;
} // namespace MeltPoolDG::Heat