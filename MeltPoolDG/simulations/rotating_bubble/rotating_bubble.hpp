#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace RotatingBubble
    {
      using namespace dealii;

      template <int dim>
      class InitializePhi : public Function<dim>
      {
      public:
        InitializePhi()
          : Function<dim>()
          , distance_sphere(dim == 1 ? Point<dim>(0.0) :
                            dim == 2 ? Point<dim>(0.0, 0.5) :
                                       Point<dim>(0.0, 0.5, 0.0),
                            0.25 /*radius*/)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
          return -distance_sphere.value(p);
        }

      private:
        const Functions::SignedDistance::Sphere<dim> distance_sphere;
      };

      template <int dim>
      class AdvectionField : public Function<dim>
      {
      public:
        AdvectionField()
          : Function<dim>(dim)
        {}

        void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
          if (dim >= 2)
            {
              const double x = p[0];
              const double y = p[1];

              values[0] = 4 * y;
              values[1] = -4 * x;
            }
          else
            AssertThrow(false, ExcMessage("Advection field for dim!=2 not implemented"));

          if (dim == 3)
            values[2] = 0;
        }
      };

      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationRotatingBubble : public SimulationBase<dim>
      {
      public:
        SimulationRotatingBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {}

        void
        create_spatial_discretization() override
        {
          if (dim == 1 || this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
            {
              AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
              this->triangulation = std::make_shared<Triangulation<dim>>();
            }
          else
            {
              this->triangulation =
                std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
            }

          if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
            {
              GridGenerator::subdivided_hyper_cube_with_simplices(
                *this->triangulation,
                Utilities::pow(2, this->parameters.base.global_refinements),
                left_domain,
                right_domain);
            }
          else
            {
              GridGenerator::subdivided_hyper_cube(*this->triangulation,
                                                   2,
                                                   left_domain,
                                                   right_domain);
              this->triangulation->refine_global(this->parameters.base.global_refinements - 1);
            }
        }

        void
        set_boundary_conditions() override
        {
          /*
           *  create a pair of (boundary_id, dirichlet_function)
           */
          constexpr types::boundary_id inflow_bc  = 42;
          constexpr types::boundary_id do_nothing = 0;

          this->attach_dirichlet_boundary_condition(
            inflow_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
          /*
           *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
           *  due to the hyperbolic nature of the analyzed problem
           *
                      out    in
          (-1,1)  +---------------+ (1,1)
                  |       :       |
            in    |       :       | out
                  |_______________|
                  |       :       |
            out   |       :       | in
                  |       :       |
                  +---------------+
           * (-1,-1)  in     out   (1,-1)
           */
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  const double half_line = (right_domain + left_domain) / 2;

                  if (face->center()[0] == left_domain && face->center()[1] > half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[0] == right_domain && face->center()[1] < half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[1] == right_domain && face->center()[0] > half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[1] == left_domain && face->center()[0] < half_line)
                    face->set_boundary_id(inflow_bc);
                  else
                    face->set_boundary_id(do_nothing);
                }
        }

        void
        set_field_conditions() override
        {
          this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "signed_distance");
          this->attach_advection_field(std::make_shared<AdvectionField<dim>>(), "level_set");
        }

      private:
        double left_domain  = -1.0;
        double right_domain = 1.0;
      };

    } // namespace RotatingBubble
  }   // namespace Simulation
} // namespace MeltPoolDG
