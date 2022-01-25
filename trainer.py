import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        #if args.gpu is not None:
        #    images = images.cuda(args.gpu, non_blocking=True)

        #target = target.cuda(args.gpu, non_blocking=True).long()
        images = images.cuda()

        target = target.cuda().long()

        # compute output
        output = model(images)

        if args.conv_type == "LCSConv":
            masks = [m.mask for m in model.mask_modules]
            entries_sum = sum(m.sum() for m in masks)
            tps = [torch.abs(m.tp) * m.mask.numel() for m in model.mask_modules]
            tps_sum = sum(m for m in tps)
            nums = [m.mask.numel() for m in model.mask_modules]
            allnums = sum(t for t in nums)
            loss = args.alpha1 * criterion(output, target.view(-1)) + args.alpha2 * tps_sum / allnums + args.lmbda * entries_sum / allnums
        else:
            loss = criterion(output, target.view(-1))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        if args.conv_type == "LCSConv":
            for optim in optimizer: optim.zero_grad()
            loss.backward()
            for optim in optimizer: optim.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch, prefix="test"):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

        if args.conv_type == "LCSConv":
            masks = [m.mask for m in model.mask_modules]
            entries_sum = sum(m.sum() for m in masks)
            tps = [torch.abs(m.tp) * m.mask.numel() for m in model.mask_modules]
            tps_sum = sum(m for m in tps)
            nums = [m.mask.numel() for m in model.mask_modules]
            allnums = sum(t for t in nums)
            loss = args.alpha1 * criterion(output, target.view(-1)) + args.alpha2 * tps_sum / allnums + args.lmbda * entries_sum / allnums
        else:
            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix=prefix, global_step=epoch)

    return top1.avg, top5.avg

