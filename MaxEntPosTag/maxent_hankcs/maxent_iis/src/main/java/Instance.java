
/**
 * ʵ��
 */
public class Instance
{
    /**
     * ��ǩ
     */
    public int label;
    /**
     * ����
     */
    public Feature feature;

    public Instance(int label, int[] xs)
    {
        this.label = label;
        this.feature = new Feature(xs);
    }

    public int getLabel()
    {
        return label;
    }

    public Feature getFeature()
    {
        return feature;
    }

    @Override
    public String toString()
    {
        return "Instance{" +
                "label=" + label +
                ", feature=" + feature +
                '}';
    }
}
