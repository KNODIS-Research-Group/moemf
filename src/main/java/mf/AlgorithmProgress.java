package mf;

import org.moeaframework.util.progress.ProgressEvent;
import org.moeaframework.util.progress.ProgressListener;

public class AlgorithmProgress implements ProgressListener {
    @Override
    public void progressUpdate(ProgressEvent event) {
        if (!event.isSeedFinished()) {
            System.out.println(
                    String.format("Evaluations: %5d/%d (%.2f %%)\tElapsed: %s\tRemaining: %s",
                            event.getCurrentNFE(),
                            event.getMaxNFE(),
                            event.getPercentComplete() * 100.0,
                            format(event.getElapsedTime()),
                            format(event.getRemainingTime()))
            );
        }
    }

    private String format(double elapsedTime) {

        int hours = (int)elapsedTime / 3600;
        int minutes = ((int)elapsedTime % 3600) / 60;
        int seconds = (int)elapsedTime % 60;

        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }
}
