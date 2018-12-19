function [neglike] = cal_neglike(true_y, prob_y)
probLims = [eps, 1-eps];

if any(prob_y(:) < probLims(1) | probLims(2) < prob_y(:))
    prob_y = max(min(mu,probLims(2)),probLims(1));
end

neglogprob = -bsxfun(@times, log(prob_y), true_y) - bsxfun(@times, log(1.0-prob_y), (1.0-true_y));
prob_y = exp(-neglogprob);
neglike = -sum(log(prob_y));
