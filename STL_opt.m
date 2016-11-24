function result = STL_opt(job_id, params)
    tic
    acc = self_taught_learning(params);
    result = 100-acc;
    toc
end