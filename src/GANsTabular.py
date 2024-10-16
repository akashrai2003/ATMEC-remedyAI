from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.sequential import PARSynthesizer
from sdv.evaluation.single_table import evaluate_quality, get_column_plot



def breast_GAN(num_rows: int):
    metadata = SingleTableMetadata()
    metadata.detect_from_csv(filepath=r'A:\Downloads\Remedy.ai\datasets\breast.csv')

    synthesizer = TVAESynthesizer.load(
        filepath=r'A:\Downloads\Remedy.ai\synthesizers\TVAESbc.pkl'
    )
    
    breast_data = synthesizer.sample(
        num_rows=num_rows,
        batch_size=100,
    )

    return breast_data

def diabetes_GAN(num_rows: int):

    synthesizer = PARSynthesizer.load(filepath=r'A:\Downloads\Remedy.ai\synthesizers\diabetes.pkl')

    diabetes_data = synthesizer.sample(
        num_sequences=num_rows,
    )

    return diabetes_data

def heart_GAN(num_rows: int):

    synthesizer = CopulaGANSynthesizer.load(filepath=r'A:\Downloads\Remedy.ai\synthesizers\CGANhd.pkl')

    heart_data = synthesizer.sample(
        num_rows=num_rows,
        batch_size=100,
    )

    return heart_data

