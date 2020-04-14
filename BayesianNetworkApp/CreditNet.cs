using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace BayesianNetworkApp
{
    public class CreditNetModel
    {
        // Primary random variables
        public VariableArray<int> Age;
        public VariableArray<int> PaymentHistory;
        public VariableArray<int> Gender;
        public VariableArray<int> Reliability;

        public VariableArray<int> Gambler;
        public VariableArray<int> RatioOfDebtsToIncome;

        public VariableArray<int> Education;
        public VariableArray<int> Income;
        public VariableArray<int> Assets;
        public VariableArray<int> FutureIncome;

        public VariableArray<int> CreditWorthiness;

        public Variable<int> NumberOfExamples;

        // Random variables representing the parameters of the distributions
        // of the primary random variables. For child variables, these are
        // in the form of conditional probability tables (CPTs)
        public Variable<Vector> ProbAge;
        public VariableArray<Vector> CPTPaymentHistory;
        public Variable<Vector> ProbGender;
        public VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]> CPTReliability;

        public Variable<Vector> ProbGambler;
        public VariableArray<Vector> CPTRatioOfDebtsToIncome;

        public Variable<Vector> ProbEducation;
        public VariableArray<Vector> CPTIncome;
        public VariableArray<Vector> CPTAssets;
        public VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]> CPTFutureIncome;

        public VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]> CPTCreditWorthiness;

        // Prior distributions for the probability and CPT variables.
        // The prior distributions are formulated as Infer.NET variables
        // so that they can be set at runtime without recompiling the model
        public Variable<Dirichlet> ProbAgePrior;
        public VariableArray<Dirichlet> CPTPaymentHistoryPrior;
        public Variable<Dirichlet> ProbGenderPrior;
        public VariableArray<VariableArray<VariableArray<Dirichlet>, Dirichlet[][]>, Dirichlet[][][]> CPTReliabilityPrior;

        public Variable<Dirichlet> ProbGamblerPrior;
        public VariableArray<Dirichlet> CPTRatioOfDebtsToIncomePrior;

        public Variable<Dirichlet> ProbEducationPrior;
        public VariableArray<Dirichlet> CPTIncomePrior;
        public VariableArray<Dirichlet> CPTAssetsPrior;
        public VariableArray<VariableArray<VariableArray<Dirichlet>, Dirichlet[][]>, Dirichlet[][][]> CPTFutureIncomePrior;

        public VariableArray<VariableArray<VariableArray<Dirichlet>, Dirichlet[][]>, Dirichlet[][][]> CPTCreditWorthinessPrior;

        // Posterior distributions for the probability and CPT variables.
        public Dirichlet ProbAgePosterior;
        public Dirichlet[] CPTPaymentHistoryPosterior;
        public Dirichlet ProbGenderPosterior;
        public Dirichlet[][][] CPTReliabilityPosterior;

        public Dirichlet ProbGamblerPosterior;
        public Dirichlet[] CPTRatioOfDebtsToIncomePosterior;

        public Dirichlet ProbEducationPosterior;
        public Dirichlet[] CPTIncomePosterior;
        public Dirichlet[] CPTAssetsPosterior;
        public Dirichlet[][][] CPTFutureIncomePosterior;

        public Dirichlet[][][] CPTCreditWorthinessPosterior;

        // Inference engine
        public InferenceEngine Engine = new InferenceEngine();

        /// <summary>
        /// Constructs a new CreditNet model
        /// </summary>
        public CreditNetModel()
        {
            // Set up the ranges
            NumberOfExamples = Variable.New<int>().Named("NofE");
            Range N = new Range(NumberOfExamples).Named("N");

            // Dimensions of model variables
            Range A = new Range(3).Named("A"); // Age
            Range PH = new Range(3).Named("PH"); // Payment History
            Range G = new Range(2).Named("G"); // Gender
            Range R = new Range(2).Named("R"); // Reliability

            Range Ga = new Range(2).Named("Ga"); // Gambler
            Range RD = new Range(2).Named("RD"); // Ratio of Debts to Income

            Range E = new Range(3).Named("E"); // Education
            Range I = new Range(3).Named("I"); // Income
            Range As = new Range(3).Named("As"); // Assets
            Range F = new Range(2).Named("F"); // Future Income

            Range CW = new Range(2).Named("CW"); // Credit-Worthiness

            // Define the priors and the parameters
            // For Age:
            ProbAgePrior = Variable.New<Dirichlet>().Named(nameof(ProbAgePrior));
            ProbAge = Variable<Vector>.Random(ProbAgePrior).Named(nameof(ProbAge));
            ProbAge.SetValueRange(A);
            // For Gender:
            ProbGenderPrior = Variable.New<Dirichlet>().Named(nameof(ProbGenderPrior));
            ProbGender = Variable<Vector>.Random(ProbGenderPrior).Named(nameof(ProbGender));
            ProbGender.SetValueRange(G);
            // For Gambler:
            ProbGamblerPrior = Variable.New<Dirichlet>().Named(nameof(ProbGamblerPrior));
            ProbGambler = Variable<Vector>.Random(ProbGamblerPrior).Named(nameof(ProbGambler));
            ProbGambler.SetValueRange(Ga);
            // For Education
            ProbEducationPrior = Variable.New<Dirichlet>().Named(nameof(ProbEducationPrior));
            ProbEducation = Variable<Vector>.Random(ProbEducationPrior).Named(nameof(ProbEducation));
            ProbEducation.SetValueRange(E);

            // Define the CPT priors for one parent children
            // Payment History probability table conditioned on Age
            CPTPaymentHistoryPrior = Variable.Array<Dirichlet>(A).Named(nameof(CPTPaymentHistoryPrior));
            CPTPaymentHistory = Variable.Array<Vector>(A).Named(nameof(CPTPaymentHistory));
            CPTPaymentHistory[A] = Variable<Vector>.Random(CPTPaymentHistoryPrior[A]);
            CPTPaymentHistory.SetValueRange(PH);
            // Ratio of Debts to Income probability table conditioned on Gambler
            CPTRatioOfDebtsToIncomePrior = Variable.Array<Dirichlet>(Ga).Named(nameof(CPTRatioOfDebtsToIncomePrior));
            CPTRatioOfDebtsToIncome = Variable.Array<Vector>(Ga).Named(nameof(CPTRatioOfDebtsToIncome));
            CPTRatioOfDebtsToIncome[Ga] = Variable<Vector>.Random(CPTRatioOfDebtsToIncomePrior[Ga]);
            CPTRatioOfDebtsToIncome.SetValueRange(RD);
            // Income probability table conditioned on Education
            CPTIncomePrior = Variable.Array<Dirichlet>(E).Named(nameof(CPTIncomePrior));
            CPTIncome = Variable.Array<Vector>(E).Named(nameof(CPTIncome));
            CPTIncome[E] = Variable<Vector>.Random(CPTIncomePrior[E]);
            CPTIncome.SetValueRange(I);
            // Assets probability table conditioned on Income
            CPTAssetsPrior = Variable.Array<Dirichlet>(I).Named(nameof(CPTAssetsPrior));
            CPTAssets = Variable.Array<Vector>(I).Named(nameof(CPTAssets));
            CPTAssets[I] = Variable<Vector>.Random(CPTAssetsPrior[I]);
            CPTAssets.SetValueRange(As);

            // Define the CPT priors for three parents children
            // Reliability probability table conditioned on Age, Gender and Payment History
            CPTReliabilityPrior = Variable.Array(Variable.Array(Variable.Array<Dirichlet>(PH), G), A).Named(nameof(CPTReliabilityPrior));
            CPTReliability = Variable.Array(Variable.Array(Variable.Array<Vector>(PH), G), A).Named(nameof(CPTReliability));
            CPTReliability[A][G][PH] = Variable<Vector>.Random(CPTReliabilityPrior[A][G][PH]);
            CPTReliability.SetValueRange(R);
            // Future Income probability table conditioned on Education, Income and Assets
            CPTFutureIncomePrior = Variable.Array(Variable.Array(Variable.Array<Dirichlet>(As), I), E).Named(nameof(CPTFutureIncomePrior));
            CPTFutureIncome = Variable.Array(Variable.Array(Variable.Array<Vector>(As), I), E).Named(nameof(CPTFutureIncome));
            CPTFutureIncome[E][I][As] = Variable<Vector>.Random(CPTFutureIncomePrior[E][I][As]);
            CPTFutureIncome.SetValueRange(F);
            // Credit-Worthiness probability table conditioned on Ratio of Debts to Income, Reliability and Future Income
            CPTCreditWorthinessPrior = Variable.Array(Variable.Array(Variable.Array<Dirichlet>(F), R), RD).Named(nameof(CPTCreditWorthinessPrior));
            CPTCreditWorthiness = Variable.Array(Variable.Array(Variable.Array<Vector>(F), R), RD).Named(nameof(CPTCreditWorthiness));
            CPTCreditWorthiness[RD][R][F] = Variable<Vector>.Random(CPTCreditWorthinessPrior[RD][R][F]);
            CPTCreditWorthiness.SetValueRange(CW);

            // Define the primary variables
            // Roots
            Age = Variable.Array<int>(N).Named(nameof(Age));
            Age[N] = Variable.Discrete(ProbAge).ForEach(N);
            Gender = Variable.Array<int>(N).Named(nameof(Gender));
            Gender[N] = Variable.Discrete(ProbGender).ForEach(N);
            Gambler = Variable.Array<int>(N).Named(nameof(Gambler));
            Gambler[N] = Variable.Discrete(ProbGambler).ForEach(N);
            Education = Variable.Array<int>(N).Named(nameof(Education));
            Education[N] = Variable.Discrete(ProbEducation).ForEach(N);
            // Single parent children
            PaymentHistory = AddChildFromOneParent(Age, CPTPaymentHistory).Named(nameof(PaymentHistory));
            RatioOfDebtsToIncome = AddChildFromOneParent(Gambler, CPTRatioOfDebtsToIncome).Named(nameof(RatioOfDebtsToIncome));
            Income = AddChildFromOneParent(Education, CPTIncome).Named(nameof(Income));
            Assets = AddChildFromOneParent(Income, CPTAssets).Named(nameof(Assets));
            // Triple parent children
            Reliability = AddChildFromThreeParents(Age, Gender, PaymentHistory, CPTReliability).Named(nameof(Reliability));
            FutureIncome = AddChildFromThreeParents(Education, Income, Assets, CPTFutureIncome).Named(nameof(FutureIncome));
            CreditWorthiness = AddChildFromThreeParents(RatioOfDebtsToIncome, Reliability, FutureIncome, CPTCreditWorthiness).Named(nameof(CreditWorthiness));
        }

        /// <summary>
        /// Helper function to set single observed value for random variable
        /// or clear the observed values
        /// </summary>
        /// <param name="observedValue"></param>
        /// <param name="randomVariable"></param>
        private void SetObservedValue(int? observedValue, VariableArray<int> randomVariable)
        {
            if (observedValue.HasValue)
                randomVariable.ObservedValue = new int[] { observedValue.Value };
            else
                randomVariable.ClearObservedValue();
        }

        /// <summary>
        /// Returns the probability of Credit-Worthiness given optional readings on
        /// age, gender, gambler, education, payment history, ratio of debts to income,
        /// income, assets, reliability and future income
        /// </summary>
        /// <param name="age"></param>
        /// <param name="gender"></param>
        /// <param name="gambler"></param>
        /// <param name="education"></param>
        /// <param name="paymentHistory"></param>
        /// <param name="ratioOfDebtsToIncome"></param>
        /// <param name="income"></param>
        /// <param name="assets"></param>
        /// <param name="reliability"></param>
        /// <param name="futureIncome"></param>
        /// <returns></returns>
        public double ProbCreditWorthiness(
            int? age,
            int? gender,
            int? gambler,
            int? education,
            int? paymentHistory,
            int? ratioOfDebtsToIncome,
            int? income,
            int? assets,
            int? reliability,
            int? futureIncome)
        {
            NumberOfExamples.ObservedValue = 1;

            SetObservedValue(age, Age);
            SetObservedValue(gender, Gender);
            SetObservedValue(gambler, Gambler);
            SetObservedValue(education, Education);
            SetObservedValue(paymentHistory, PaymentHistory);
            SetObservedValue(ratioOfDebtsToIncome, RatioOfDebtsToIncome);
            SetObservedValue(income, Income);
            SetObservedValue(assets, Assets);
            SetObservedValue(reliability, Reliability);
            SetObservedValue(futureIncome, FutureIncome);

            CreditWorthiness.ClearObservedValue();

            // Inference
            var creditWorthinessPosterior = Engine.Infer<Discrete[]>(CreditWorthiness);

            // index 1 is worthy and index 0 is unworthy probability
            return creditWorthinessPosterior[0].GetProbs()[1];
        }

        /// <summary>
        /// Sets prior probabilities for all parameters
        /// </summary>
        /// <param name="probAge"></param>
        /// <param name="probGender"></param>
        /// <param name="probGambler"></param>
        /// <param name="probEducation"></param>
        /// <param name="cptPaymentHistory"></param>
        /// <param name="cptRatioOfDebtsToIncome"></param>
        /// <param name="cptIncome"></param>
        /// <param name="cptAssets"></param>
        /// <param name="cptReliability"></param>
        /// <param name="cptFutureIncome"></param>
        /// <param name="cptCreditWorthiness"></param>
        /// <returns></returns>
        public void SetPriorProbabilities(
            Vector probAge,
            Vector probGender,
            Vector probGambler,
            Vector probEducation,
            Vector[] cptPaymentHistory,
            Vector[] cptRatioOfDebtsToIncome,
            Vector[] cptIncome,
            Vector[] cptAssets,
            Vector[][][] cptReliability,
            Vector[][][] cptFutureIncome,
            Vector[][][] cptCreditWorthiness)
        {
            ProbAgePrior.ObservedValue = Dirichlet.PointMass(probAge);
            ProbGenderPrior.ObservedValue = Dirichlet.PointMass(probGender);
            ProbGamblerPrior.ObservedValue = Dirichlet.PointMass(probGambler);
            ProbEducationPrior.ObservedValue = Dirichlet.PointMass(probEducation);
            CPTPaymentHistoryPrior.ObservedValue = cptPaymentHistory.Select(v => Dirichlet.PointMass(v)).ToArray();
            CPTRatioOfDebtsToIncomePrior.ObservedValue = cptRatioOfDebtsToIncome.Select(v => Dirichlet.PointMass(v)).ToArray();
            CPTIncomePrior.ObservedValue = cptIncome.Select(v => Dirichlet.PointMass(v)).ToArray();
            CPTAssetsPrior.ObservedValue = cptAssets.Select(v => Dirichlet.PointMass(v)).ToArray();
            CPTReliabilityPrior.ObservedValue = cptReliability.Select(vaa => vaa.Select(va => va.Select(v => Dirichlet.PointMass(v)).ToArray()).ToArray()).ToArray();
            CPTFutureIncomePrior.ObservedValue = cptFutureIncome.Select(vaa => vaa.Select(va => va.Select(v => Dirichlet.PointMass(v)).ToArray()).ToArray()).ToArray();
            CPTCreditWorthinessPrior.ObservedValue = cptCreditWorthiness.Select(vaa => vaa.Select(va => va.Select(v => Dirichlet.PointMass(v)).ToArray()).ToArray()).ToArray();
        }

        /// <summary>
        /// Helper method to add a child from one parent
        /// </summary>
        /// <param name="parent">Parent (a variable array over a range of examples)</param>
        /// <param name="cpt">Conditional probability table</param>
        /// <returns></returns>
        public static VariableArray<int> AddChildFromOneParent(
            VariableArray<int> parent,
            VariableArray<Vector> cpt)
        {
            var n = parent.Range;
            var child = Variable.Array<int>(n);
            using (Variable.ForEach(n))
            using (Variable.Switch(parent[n]))
            {
                child[n] = Variable.Discrete(cpt[parent[n]]);
            }

            return child;
        }

        /// <summary>
        /// Helper method to add a child from three parents
        /// </summary>
        /// <param name="parent1">First parent (a variable array over a range of examples)</param>
        /// <param name="parent2">Second parent (a variable array over the same range)</param>
        /// <param name="parent3">Third parent (a variable array over the same range)</param>
        /// <param name="cpt">Conditional probability table</param>
        /// <returns></returns>
        public static VariableArray<int> AddChildFromThreeParents(
            VariableArray<int> parent1,
            VariableArray<int> parent2,
            VariableArray<int> parent3,
            VariableArray<VariableArray<VariableArray<Vector>, Vector[][]>, Vector[][][]> cpt)
        {
            var n = parent1.Range;
            var child = Variable.Array<int>(n);
            using (Variable.ForEach(n))
            using (Variable.Switch(parent1[n]))
            using (Variable.Switch(parent2[n]))
            using (Variable.Switch(parent3[n]))
            {
                child[n] = Variable.Discrete(cpt[parent1[n]][parent2[n]][parent3[n]]);
            }

            return child;
        }
    }

    public static class CreditNet
    {
        private static CreditNetModel CreateModelWithPriorProbabilities()
        {
            // Create a new model
            CreditNetModel model = new CreditNetModel();

            // Setting assumed parameters
            Vector probAge = Vector.FromArray(0.35, 0.5, 0.15);
            Vector probGender = Vector.FromArray(0.5, 0.5);
            Vector probGambler = Vector.FromArray(0.3, 0.7);
            Vector probEducation = Vector.FromArray(0.1, 0.6, 0.3);
            Vector[] cptPaymentHistory = new Vector[]
            {
                Vector.FromArray(0.5, 0.4, 0.1), // Young
                Vector.FromArray(0.2, 0.5, 0.3), // Middle
                Vector.FromArray(0.1, 0.4, 0.5) // Old
            };
            Vector[] cptRatioOfDebtsToIncome = new Vector[]
            {
                Vector.FromArray(0.1, 0.9), // Is Gambler
                Vector.FromArray(0.7, 0.3) // Not Gambler
            };
            Vector[] cptIncome = new Vector[]
            {
                Vector.FromArray(0.75, 0.2, 0.05), // Elementary
                Vector.FromArray(0.2, 0.6, 0.2), // Secondary
                Vector.FromArray(0.1, 0.5, 0.4) // Higher
            };
            Vector[] cptAssets = new Vector[]
            {
                Vector.FromArray(0.8, 0.15, 0.05), // Low Income
                Vector.FromArray(0.3, 0.6, 0.1), // Medium Income
                Vector.FromArray(0.1, 0.4, 0.5) // High Income
            };
            Vector[][][] cptReliability = new Vector[][][]
            {
                new Vector[][] // Young
                {
                    new Vector[] // Male
                    {
                        Vector.FromArray(0.9, 0.1), // Unacceptable
                        Vector.FromArray(0.2, 0.8), // Acceptable
                        Vector.FromArray(0.05, 0.95) // Excellent
                    },
                    new Vector[] // Female
                    {
                        Vector.FromArray(0.8, 0.2), // Unacceptable
                        Vector.FromArray(0.1, 0.9), // Acceptable
                        Vector.FromArray(0.05, 0.95) // Excellent
                    }
                },
                new Vector[][] // Middle
                {
                    new Vector[] // Male
                    {
                        Vector.FromArray(0.9, 0.1), // Unacceptable
                        Vector.FromArray(0.3, 0.7), // Acceptable
                        Vector.FromArray(0.2, 0.8) // Excellent
                    },
                    new Vector[] // Female
                    {
                        Vector.FromArray(0.7, 0.3), // Unacceptable
                        Vector.FromArray(0.2, 0.8), // Acceptable
                        Vector.FromArray(0.05, 0.95) // Excellent
                    }
                },
                new Vector[][] // Old
                {
                    new Vector[] // Male
                    {
                        Vector.FromArray(0.8, 0.2), // Unacceptable
                        Vector.FromArray(0.2, 0.8), // Acceptable
                        Vector.FromArray(0.1, 0.9) // Excellent
                    },
                    new Vector[] // Female
                    {
                        Vector.FromArray(0.6, 0.4), // Unacceptable
                        Vector.FromArray(0.15, 0.85), // Acceptable
                        Vector.FromArray(0.01, 0.99) // Excellent
                    }
                }
            };
            Vector[][][] cptFutureIncome = new Vector[][][]
            {
                new Vector[][] // Elementary
                {
                    new Vector[] // Low Income
                    {
                        Vector.FromArray(0.99, 0.01), // Low Assets
                        Vector.FromArray(0.88, 0.12), // Medium Assets
                        Vector.FromArray(0.8, 0.2) // High Assets
                    },
                    new Vector[] // Medium Income
                    {
                        Vector.FromArray(0.85, 0.15), // Low Assets
                        Vector.FromArray(0.75, 0.25), // Medium Assets
                        Vector.FromArray(0.7, 0.3) // High Assets
                    },
                    new Vector[] // High Income
                    {
                        Vector.FromArray(0.8, 0.2), // Low Assets
                        Vector.FromArray(0.7, 0.3), // Medium Assets
                        Vector.FromArray(0.6, 0.4) // High Assets
                    }
                },
                new Vector[][] // Secondary
                {
                    new Vector[] // Low Income
                    {
                        Vector.FromArray(0.9, 0.1), // Low Assets
                        Vector.FromArray(0.8, 0.2), // Medium Assets
                        Vector.FromArray(0.7, 0.3) // High Assets
                    },
                    new Vector[] // Medium Income
                    {
                        Vector.FromArray(0.8, 0.2), // Low Assets
                        Vector.FromArray(0.7, 0.3), // Medium Assets
                        Vector.FromArray(0.6, 0.4) // High Assets
                    },
                    new Vector[] // High Income
                    {
                        Vector.FromArray(0.7, 0.3), // Low Assets
                        Vector.FromArray(0.6, 0.4), // Medium Assets
                        Vector.FromArray(0.5, 0.5) // High Assets
                    }
                },
                new Vector[][] // Higher
                {
                    new Vector[] // Low Income
                    {
                        Vector.FromArray(0.3, 0.7), // Low Assets
                        Vector.FromArray(0.4, 0.6), // Medium Assets
                        Vector.FromArray(0.5, 0.5) // High Assets
                    },
                    new Vector[] // Medium Income
                    {
                        Vector.FromArray(0.3, 0.7), // Low Assets
                        Vector.FromArray(0.2, 0.8), // Medium Assets
                        Vector.FromArray(0.1, 0.9) // High Assets
                    },
                    new Vector[] // High Income
                    {
                        Vector.FromArray(0.2, 0.8), // Low Assets
                        Vector.FromArray(0.1, 0.9), // Medium Assets
                        Vector.FromArray(0.05, 0.95) // High Assets
                    }
                }
            };
            Vector[][][] cptCreditWorthiness = new Vector[][][]
            {
                new Vector[][] // Low Ratio of Debts to Income
                {
                    new Vector[] // Unreliable
                    {
                        Vector.FromArray(0.4, 0.6), // Low Future Income
                        Vector.FromArray(0.2, 0.8) // High Future Income
                    },
                    new Vector[] // Reliable
                    {
                        Vector.FromArray(0.3, 0.7), // Low Future Income
                        Vector.FromArray(0.01, 0.99) // High Future Income
                    }
                },
                new Vector[][] // High Ratio of Debts to Income
                {
                    new Vector[] // Unreliable
                    {
                        Vector.FromArray(0.99, 0.01), // Low Future Income
                        Vector.FromArray(0.7, 0.3) // High Future Income
                    },
                    new Vector[] // Reliable
                    {
                        Vector.FromArray(0.8, 0.2), // Low Future Income
                        Vector.FromArray(0.6, 0.4) // High Future Income
                    }
                }
            };

            model.SetPriorProbabilities(probAge, probGender, probGambler, probEducation, cptPaymentHistory,
                cptRatioOfDebtsToIncome, cptIncome, cptAssets, cptReliability, cptFutureIncome, cptCreditWorthiness);

            return model;
        }

        private static void ResetConsoleColor()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine();
        }
        
        public static void CountCreditWorthinessProbability()
        {
            var model = CreateModelWithPriorProbabilities();

            while (true)
            {
                Console.WriteLine("Enter parameters (skip = only comma): age (0-2), gender (0=M, 1=F), gambler (0=T, 1=F),");
                Console.WriteLine("education (0-2), payment history (0=U, 1=A, 2=E), ratio of debts to income (0=L, 1=H),");
                Console.WriteLine("income (0-2), assets (0-2), reliability (0=U, 1=R), future income (0=L, 1=H)");
                Console.WriteLine("(seperated by comma in a single line and press Enter or simply press Enter to exit)");
                Console.WriteLine();

                var args = Console.ReadLine();
                Console.WriteLine();

                if (string.IsNullOrEmpty(args))
                    return;
                else
                {
                    try
                    {
                        var parameters = args.Split(",").Select(p => string.IsNullOrEmpty(p) ? null : (int?)int.Parse(p)).ToArray();
                        int? age = parameters[0];
                        int? gender = parameters[1];
                        int? gambler = parameters[2];
                        int? education = parameters[3];
                        int? paymentHistory = parameters[4];
                        int? ratioOfDebtsToIncome = parameters[5];
                        int? income = parameters[6];
                        int? assets = parameters[7];
                        int? reliability = parameters[8];
                        int? futureIncome = parameters[9];

                        double probability = model.ProbCreditWorthiness(age, gender, gambler, education, paymentHistory,
                            ratioOfDebtsToIncome, income, assets, reliability, futureIncome);

                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine("P(credit-worthiness | parameter values) = {0:0.0000}", probability);
                        ResetConsoleColor();
                    }
                    catch (Exception)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine("Wrong set of parameters!");
                        ResetConsoleColor();
                    }
                }
            }
        }
    }
}
