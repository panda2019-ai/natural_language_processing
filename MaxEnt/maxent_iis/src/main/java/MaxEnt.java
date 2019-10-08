import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * ??????IIS??Improved Iterative Scaling???????
 * User: tpeng  <pengtaoo@gmail.com>
 */
public class MaxEnt
{

    private final static boolean DEBUG = false;

    /**
     * ????????
     */
    private final int ITERATIONS = 200;

    /**
     * ??????????
     */
    private static final double EPSILON = 0.001;

    // the number of training instances
    /**
     * ????????
     */
    private int N;

    // the minimal of Y
    /**
     * Y??????
     */
    private int minY;

    // the maximum of Y
    /**
     * Y??????
     */
    private int maxY;

    // the empirical expectation value of f(x, y)
    /**
     * ????????????
     */
    private double empirical_expects[];

    // the weight to learn.
    /**
     * ??????
     */
    private double w[];

    /**
     * ??????
     */
    private List<Instance> instances = new ArrayList<Instance>();

    /**
     * ???????????
     */
    private List<FeatureFunction> functions = new ArrayList<FeatureFunction>();

    /**
     * ???????
     */
    private List<Feature> features = new ArrayList<Feature>();

    public static void main(String... args) throws FileNotFoundException
    {
        List<Instance> instances = DataSet.readDataSet("examples/zoo.train");
        MaxEnt me = new MaxEnt(instances);
        me.train();

        List<Instance> trainInstances = DataSet.readDataSet("examples/zoo.test");
        int pass = 0;
        for (Instance instance : trainInstances)
        {
            int predict = me.classify(instance);
            if (predict == instance.getLabel())
            {
                pass += 1;
            }
        }

        System.out.println("accuracy: " + 1.0 * pass / trainInstances.size());
    }

    public MaxEnt(List<Instance> trainInstance)
    {

        instances.addAll(trainInstance);
        N = instances.size();
        createFeatFunctions(instances);
        w = new double[functions.size()];
        empirical_expects = new double[functions.size()];
        calc_empirical_expects();
    }

    /**
     * ????????????
     * @param instances ???
     */
    private void createFeatFunctions(List<Instance> instances)
    {

        int maxLabel = 0;
        int minLabel = Integer.MAX_VALUE;
        int[] maxFeatures = new int[instances.get(0).getFeature().getValues().length];
        LinkedHashSet<Feature> featureSet = new LinkedHashSet<Feature>();

        for (Instance instance : instances)
        {

            if (instance.getLabel() > maxLabel)
            {
                maxLabel = instance.getLabel();
            }
            if (instance.getLabel() < minLabel)
            {
                minLabel = instance.getLabel();
            }


            for (int i = 0; i < instance.getFeature().getValues().length; i++)
            {
                if (instance.getFeature().getValues()[i] > maxFeatures[i])
                {
                    maxFeatures[i] = instance.getFeature().getValues()[i];
                }
            }

            featureSet.add(instance.getFeature());
        }

        features = new ArrayList<Feature>(featureSet);

        maxY = maxLabel;
        minY = minLabel;

        for (int i = 0; i < maxFeatures.length; i++)
        {
            for (int x = 0; x <= maxFeatures[i]; x++)
            {
                for (int y = minY; y <= maxLabel; y++)
                {
                    functions.add(new FeatureFunction(i, x, y));
                }
            }
        }

        if (DEBUG)
        {
            System.out.println("# features = " + features.size());
            System.out.println("# functions = " + functions.size());
        }
    }

    // calculates the p(y|x)
    /**
     * ???????????? p(y|x)
     * @return p(y|x)
     */
    private double[][] calc_prob_y_given_x()
    {

        double[][] cond_prob = new double[features.size()][maxY + 1];

        for (int y = minY; y <= maxY; y++)
        {
            for (int i = 0; i < features.size(); i++)
            {
                double z = 0;
                for (int j = 0; j < functions.size(); j++)
                {
                    z += w[j] * functions.get(j).apply(features.get(i), y);
                }
                cond_prob[i][y] = Math.exp(z);
            }
        }

        for (int i = 0; i < features.size(); i++)
        {
            double normalize = 0;
            for (int y = minY; y <= maxY; y++)
            {
                normalize += cond_prob[i][y];
            }
            for (int y = minY; y <= maxY; y++)
            {
                cond_prob[i][y] /= normalize;
            }
        }

        return cond_prob;
    }

    /**
     * ???
     */
    public void train()
    {
        for (int k = 0; k < ITERATIONS; k++)
        {
            for (int i = 0; i < functions.size(); i++)
            {
                double delta = iis_solve_delta(empirical_expects[i], i);
                w[i] += delta;
            }
            if (DEBUG)  System.out.println("ITERATIONS: " + k + " " + Arrays.toString(w));
        }
    }

    /**
     * ????
     * @param instance
     * @return
     */
    public int classify(Instance instance)
    {

        double max = 0;
        int label = 0;

        for (int y = minY; y <= maxY; y++)
        {
            double sum = 0;
            for (int i = 0; i < functions.size(); i++)
            {
                sum += Math.exp(w[i] * functions.get(i).apply(instance.getFeature(), y));
            }
            if (sum > max)
            {
                max = sum;
                label = y;
            }
        }
        return label;
    }

    /**
     * ??????????
     */
    private void calc_empirical_expects()
    {

        for (Instance instance : instances)
        {
            int y = instance.getLabel();
            Feature feature = instance.getFeature();
            for (int i = 0; i < functions.size(); i++)
            {
                empirical_expects[i] += functions.get(i).apply(feature, y);
            }
        }
        for (int i = 0; i < functions.size(); i++)
        {
            empirical_expects[i] /= 1.0 * N;
        }
        if (DEBUG)  System.out.println(Arrays.toString(empirical_expects));
    }

    /**
     * ???????????????????????
     * @param feature
     * @param y
     * @return
     */
    private int apply_f_sharp(Feature feature, int y)
    {

        int sum = 0;
        for (int i = 0; i < functions.size(); i++)
        {
            FeatureFunction function = functions.get(i);
            sum += function.apply(feature, y);
        }
        return sum;
    }

    /**
     * ??delta_i
     * @param empirical_e fi??????
     * @param fi fi???±?
     * @return delta_i
     */
    private double iis_solve_delta(double empirical_e, int fi)
    {

        double delta = 0;
        double f_newton, df_newton;
        double p_yx[][] = calc_prob_y_given_x();

        int iters = 0;

        while (iters < 50)                                  // ????
        {
            f_newton = df_newton = 0;
            for (int i = 0; i < instances.size(); i++)
            {
                Instance instance = instances.get(i);
                Feature feature = instance.getFeature();
                int index = features.indexOf(feature);
                for (int y = minY; y <= maxY; y++)
                {
                    int f_sharp = apply_f_sharp(feature, y);
                    double prod = p_yx[index][y] * functions.get(fi).apply(feature, y) * Math.exp(delta * f_sharp);
                    f_newton += prod;
                    df_newton += prod * f_sharp;
                }
            }
            f_newton = empirical_e - f_newton / N;      // g
            df_newton = -df_newton / N;                 // g?????

            if (Math.abs(f_newton) < 0.0000001)
                return delta;

            double ratio = f_newton / df_newton;

            delta -= ratio;
            if (Math.abs(ratio) < EPSILON)
            {
                return delta;
            }
            iters++;
        }
        throw new RuntimeException("IIS did not converge"); // w_i??????
    }

    /**
     * ????????
     */
    class FeatureFunction
    {

        private int index;
        private int value;
        private int label;

        FeatureFunction(int index, int value, int label)
        {
            this.index = index;
            this.value = value;
            this.label = label;
        }

        /**
         * ??????
         * @param feature ????X?????????????index?????
         * @param label Y
         * @return
         */
        public int apply(Feature feature, int label)
        {
            if (feature.getValues()[index] == value && label == this.label)
                return 1;
            return 0;
        }
    }
}


