-- Run in Supabase SQL Editor while counter.py and sync_counts_to_supabase.py
-- are stopped. Replace target_source_file with the value printed below.
begin;

do $$
declare
    target_source_file text := '2026-08-12/count_20260812.csv';
    previous_maximum bigint;
    target_minimum bigint;
begin
    lock table public.traffic_counts in access exclusive mode;
    select coalesce(max(record_id), 0) into previous_maximum
    from public.traffic_counts where source_file <> target_source_file;
    select min(record_id) into target_minimum
    from public.traffic_counts where source_file = target_source_file;

    if target_minimum is not null and target_minimum <= previous_maximum then
        update public.traffic_counts
        set record_id = record_id + previous_maximum
        where source_file = target_source_file;
    end if;
end $$;

alter table public.traffic_counts drop constraint if exists traffic_counts_pkey;
alter table public.traffic_counts add primary key (record_id);
create index if not exists traffic_counts_source_file_index on public.traffic_counts (source_file);
commit;
