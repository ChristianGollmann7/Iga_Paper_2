/// 
#include <gismo.h>

//! [Includes]
#ifdef gsElasticity_ENABLED
#include <gsElasticity/src/gsElasticityAssembler.h>
#include <gsElasticity/src/gsWriteParaviewMultiPhysics.h>
#include <gsElasticity/src/gsGeoUtils.h>
#endif

#include <gsStructuralAnalysis/src/gsStaticSolvers/gsStaticNewton.h>
#include <gsStructuralAnalysis/src/gsStructuralAnalysisTools/gsStructuralAnalysisUtils.h>
//! [Includes]

#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace gismo;

int main(int argc, char* argv[]){

    //=====================================//
        // This simulates a geometry being fixated on the south-side and being stretched on the north-side.
        // West- and east-side are being traction free.
    //=====================================//
                // Input //
    //=====================================//

    std::string filename;
    real_t youngsModulus = 1;
    real_t poissonsRatio = 0.3;
    index_t numUniRef = 0;
    index_t numDegElev = 0;
    index_t numPlotPoints = 2000;
    real_t x_displ = 2.0;
    real_t y_displ = 0.0;
    real_t quA = 1.0;
    index_t quB = 1;
    index_t testCase = 0; // 0: UAT, 1: fixed sides
    std::string filePath = "";
    real_t relaxation = 0.05;
    real_t angle = 0.0;


    // minimalistic user interface for terminal
    gsCmdLine cmd("Geometry being stretched with nonlinear elasticity solver.");
    cmd.addReal("p","poisson","Poisson's ratio used in the material law",poissonsRatio);
    cmd.addReal("E","Young","Young's modulus used in the material law",youngsModulus);
    cmd.addReal("x","x_displacement","displacement in x direction",x_displ);
    cmd.addReal("y","y_displacement","displacement in y direction",y_displ);
    cmd.addInt("r","refine","Number of uniform refinement application",numUniRef);
    cmd.addInt("d","degelev","Number of degree elevation application",numDegElev);
    cmd.addInt("s","point","Number of points to plot to Paraview",numPlotPoints);
    cmd.addReal("A","quA","Number of quadrature points: quA*deg + quB",quA);
    cmd.addInt("B","quB","Number of quadrature points: quA*deg + quB",quB);
    cmd.addString("P", "path", "Path for saving the output file", filePath);
    cmd.addString("f", "input_geometry", "Path of input geometry file", filename);
    cmd.addInt("t","testCase","testCase",testCase);
    cmd.addReal("R","relaxation","relaxation parameter of Newton method",relaxation);
    cmd.addReal("a","angle","rotation angle for upper side",angle);
    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    //=============================================//
        // Scanning geometry and creating bases //
    //=============================================//

    // scanning geometry
    gsMultiPatch<> geometry;
    gsReadFile<>(filename, geometry);

    // creating bases
    gsMultiBasis<> basisDisplacement(geometry);

    for (index_t i = 0; i < numDegElev; ++i)
        basisDisplacement.degreeElevate();
    for (index_t i = 0; i < numUniRef; ++i)
        basisDisplacement.uniformRefine();


    //=============================================//
        // Setting loads and boundary conditions //
    //=============================================//

    // BC
    gsConstantFunction<> f(0.,0.,2);
    // source function, rhs
    gsConstantFunction<> g(0.,0.,2);
    
    // build expressions for the boundary conditions, including the rotation of upper side
    real_t ca = std::cos(angle);
    real_t sa = std::sin(angle);
    std::string x_expression = std::to_string(ca) + "*x - " + std::to_string(sa) + "*y - x + " + std::to_string(x_displ);
    std::string y_expression = std::to_string(sa) + "*x + " + std::to_string(ca) + "*y - y + " + std::to_string(y_displ);
    //gsConstantFunction<> displ_x(x_displ, 2);
    //gsConstantFunction<> displ_y(y_displ, 2);
    gsFunctionExpr<real_t> displ_x(x_expression, 2);
    gsFunctionExpr<real_t> displ_y(y_expression, 2);
    // boundary conditions
    gsBoundaryConditions<> bcInfo;
    bcInfo.setGeoMap(geometry); 

    // neumann BC
    gsConstantFunction<> nm(0.,0., 2);

    bcInfo.addCondition(0,boundary::west,condition_type::dirichlet,nullptr,0);
    bcInfo.addCondition(0,boundary::west,condition_type::dirichlet,nullptr,1);
    bcInfo.addCondition(0,boundary::east,condition_type::dirichlet,&displ_x,0);
    bcInfo.addCondition(0,boundary::east,condition_type::dirichlet,&displ_y,1);
    bcInfo.addCondition(0,boundary::north,condition_type::neumann,&nm);
    bcInfo.addCondition(0,boundary::south,condition_type::neumann,&nm);

    //=============================================//
                  // Solving //
    //=============================================//
    
    gsInfo << "Solving...\n";
    gsStopwatch clock;
    clock.restart();

    // creating assembler
    gsElasticityAssembler<real_t> assembler(geometry,basisDisplacement,bcInfo,g);
    assembler.options().setReal("YoungsModulus",youngsModulus);
    assembler.options().setReal("PoissonsRatio",poissonsRatio);
    assembler.options().setReal("quA",quA);
    assembler.options().setInt("quB",quB);
    assembler.options().setInt("MaterialLaw",material_law::neo_hooke_ln);
    assembler.options().setInt("DirichletValues",dirichlet::l2Projection);
    //assembler.options().setInt("DirichletValues",dirichlet::penalize);

    //! [Define nonlinear residual functions]
    std::vector<gsMatrix<> > fixedDofs = assembler.allFixedDofs();
    // Function for the Jacobian
    gsStructuralAnalysisOps<real_t>::Jacobian_t Jacobian = [&assembler,&fixedDofs](gsVector<real_t> const &x, gsSparseMatrix<real_t> & m)
    {
        assembler.homogenizeFixedDofs(-1);
        assembler.assemble(x,fixedDofs);
        m = assembler.matrix();
        return true;
    };

    // Function for the Residual
    gsStructuralAnalysisOps<real_t>::Residual_t Residual = [&fixedDofs,&assembler](gsVector<real_t> const &x, gsVector<real_t> & v)
    {
        assembler.homogenizeFixedDofs(-1);
        assembler.assemble(x,fixedDofs);
        v = assembler.rhs();
        return true;
    };
    //! [Define nonlinear residual functions]

    //! [Assemble linear part]
    assembler.assemble();
    gsSparseMatrix<> K = assembler.matrix();
    gsVector<> F = assembler.rhs();
    //! [Assemble linear part]

    //! [Set static solver]
    gsStaticNewton<real_t> solver(K,F,Jacobian,Residual);
    solver.options().setInt("verbose",1);
    solver.options().addReal("Relaxation","Relaxation parameter",relaxation);
    solver.options().addReal("tol","Relative Tolerance",1e-9);
    solver.options().setInt("maxIt", 5000);
    solver.initialize();
    //! [Set static solver]
    
    //! [Solve nonlinear problem]
    gsInfo<<"Solving system with "<<assembler.numDofs()<<" DoFs\n";
    gsStatus status = solver.solve();
    GISMO_ASSERT(status==gsStatus::Success,"Solver failed");
    gsVector<> solVector = solver.solution();
    //! [Solve nonlinear problem]

    real_t solution_time = clock.stop();
    gsInfo << "Solved the system in " << solution_time <<"s.\n";


    //=============================================//
                  // Output //
    //=============================================//

    // solution to the nonlinear problem as an isogeometric displacement field
    assembler.setFixedDofs(fixedDofs);
    gsMultiPatch<> displacement;
    assembler.constructSolution(solVector, fixedDofs, displacement);
    gsPiecewiseFunction<> stresses;
    assembler.constructCauchyStresses(displacement,stresses,stress_components::von_mises);

 
    if (numPlotPoints > 0)
    {
        // constructing an IGA field (geometry + solution)
        gsField<> displacementField(assembler.patches(),displacement);
        gsField<> stressField(assembler.patches(),stresses,true);
        // creating a container to plot all fields to one Paraview file
        std::map<std::string,const gsField<> *> fields;
        fields["Displacement"] = &displacementField;
        fields["von Mises"] = &stressField;
        gsWriteParaviewMultiPhysics(fields,"Experiment_4_nonlinear",numPlotPoints);
        gsInfo << "Open \"Experiment_4_nonlinear.pvd\" in Paraview for visualization.\n";
    }
    

    // Write solution to file
    std::string geometry_name = filePath + "geometry_gismo";
    gsWrite(geometry, geometry_name.c_str());
    std::string displacement_name = filePath + "displacement_gismo";
    gsWrite(displacement, displacement_name.c_str());
    // Write out the simulation duration
    std::string time_name = filePath + "runtimes.csv";
    std::ofstream outFile(time_name, std::ios_base::app);
    outFile << numUniRef << ", " << numDegElev+1 << ", " << solution_time << std::endl;
    


    return 0;
}
